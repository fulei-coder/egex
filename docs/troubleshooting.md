# 问题排查

> 本文档记录实际部署中**具有一定隐蔽性**的问题。基础连接检查（ping、USB 识别、SteamVR 状态）不在此列。

---

## 采集阶段

### 相机 "Device is already in use"

**原因**：RealSense 设备锁未释放（上次脚本非正常退出），或两个 D435i 接在同一 USB 控制器上带宽不足。

```bash
pkill -f realsense && pkill -f collect_data
# 反复出现 → lsusb -t 确认两相机是否分属不同 USB 控制器
```

后者更隐蔽——报错信息和设备占用完全一样，但 `pkill` 无效。`lsusb -t` 看树形拓扑即可定位。

---

## 训练阶段

### CUDA OOM

**按优先级处理**：

```yaml
# 1. 减 batch_size
batch_size: 4

# 2. 减 num_workers（释放共享内存，防 Bus error）
num_workers: 1

# 3. VLA 策略开启冻结 + 梯度检查点
policy:
  freeze_vision_encoder: true
  train_expert_only: true
  gradient_checkpointing: true
```

**显存参考值**（单卡）：

| 策略 | RTX 4080 12GB | A800 80GB |
|------|--------------|-----------|
| ACT | batch ≤ 64 | batch ≤ 128 |
| Diffusion | batch ≤ 32 | batch ≤ 128 |
| SmolVLA | batch ≤ 4 | batch ≤ 16 |
| Pi0 | ❌ | batch ≤ 2（需 gradient_checkpointing） |

### Bus error（非 GPU OOM）

**原因**：`/dev/shm`（共享内存）被 DataLoader `num_workers` 撑满，不是显存问题。报错堆栈里没有 CUDA 关键词。

```bash
df -h /dev/shm  # 检查共享内存
```

Docker 用户注意 `--shm-size` 默认只有 64MB。

### Loss 不下降

| 现象 | 原因 | 解决 |
|------|------|------|
| 完全平的 | lr 太小 / DataLoader 返回空数据 | 调大 lr / 打印一个 batch 检查 |
| 剧烈震荡 | lr 太大 / batch 太小 | 调小 lr / 加大 batch |
| 缓慢下降后停在高位 | 训练量不够 | 增加 steps 或数据量 |

先排除数据问题：
```bash
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lerobot/pick_cube_30fps', root='data/pick_cube_30fps')
print(f'FPS: {ds.fps}, 总帧数: {len(ds)}, Episode数: {ds.num_episodes}')
"
```

### VQ-BeT 训练极慢（IO 瓶颈）

**原因**：`n_obs_steps=5` 时每个样本需解码 5 帧视频，CPU 成为瓶颈——GPU 利用率低但 CPU 打满。

**解决**：
- 降 `num_workers`（多 worker 争抢 CPU 解码反而更慢）
- 使用 `video_backend: pyav`
- 小数据集可预解码为图片格式绕过

### 断点续训参数解析

```bash
# ❌ 空格连接会被 argparse 拆分为两个参数
--config_path outputs/xxx/train_config.json

# ✅ 等号连接
--config_path=outputs/xxx/checkpoints/last/pretrained_model/train_config.json
--resume=true
```

`--resume` 是 bool，配置恢复靠 `--config_path` 指向 checkpoint 内的 `train_config.json`。

### lr preset 静默覆盖（VLA 策略）

**现象**：yaml 里配了 `lr: 2e-5`，但实际训练用的是另一个值——WandB 日志里 lr 和配置文件对不上。

**原因**：LeRobot VLA 策略有 `use_policy_training_preset=True`，`validate()` 时会**用 policy 内置 preset 覆盖 yaml 中的 optimizer 参数**。

**解决**：
```yaml
use_policy_training_preset: false
```
或直接修改 `configuration_pi0.py` / `configuration_smolvla.py` 里的 preset 默认值。

---

## 推理阶段

### 输出值接近零 / 机械臂不动

**原因**：preprocessor/postprocessor 加载失败。checkpoint 目录缺少 `policy_preprocessor.json` 时，脚本**静默跳过**归一化还原 → 模型输出的归一化值（接近 0）直接发给机械臂 → 关节角变化极小。

**验证**：打印模型原始输出，检查值域是否在 [-1, 1] 附近（未还原）还是在关节角真实范围内（已还原）。

### 动作幅度 / 速度异常

**原因**：`--freq` 和训练数据 FPS 不一致。BC 学的是 `a_t = s_{t+1}`（1/FPS 秒后的状态），推理频率不同等于改变时间尺度。

```bash
# 检查训练数据 FPS
python -c "
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset('lerobot/pick_cube_30fps', root='data/pick_cube_30fps')
print('训练数据 FPS:', ds.fps)  # --freq 应与此一致
"
```

### 推理频率骤降至 1-2Hz（Modbus 延迟）

**现象**：预期 30Hz，实际 1-2Hz。排查时容易误判为模型推理瓶颈——在 GPU 侧优化半天无效。加 timing 日志后发现：`obs_time` 远大于 `model_time`。

**原因**：Modbus RTU 9600 波特率下，同步读写含超时等待，单次 300-500ms。放在主循环中直接成为瓶颈。

**解决**：
```python
# 异步写入（不阻塞主循环）
if gripper_changed:
    threading.Thread(target=write_gripper, args=(value,), daemon=True).start()

# 缓存读取——夹爪只有开/闭状态，不需要每帧读取
```

效果：1-2Hz → 30Hz+。

> 经验：实机系统的性能瓶颈往往不在 GPU 推理，而在硬件通讯环节。Modbus/CAN/串口都可能有类似的超时机制。

---

## VLA 模型

### Gated 模型下载 403

Pi0 依赖的 PaliGemma、SmolVLA 依赖的 SmolVLM 都是 gated model，需要在 HuggingFace 页面 "Agree and access" 后才能下载。

### 离线环境（集群）

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/path/to/cache  # TRANSFORMERS_CACHE 已废弃
```

在有网环境预下载：`AutoModel.from_pretrained(...)` 一次即可缓存。

### VLA 配置版本不兼容

**现象**：`unexpected keyword argument`（如 `use_dual_state_input`）。

**原因**：SmolVLA 不同版本间配置字段有变化，旧 checkpoint 的 config 含新版本已移除的字段。

**解决**：推理脚本中 `_patch_config_compat` 做兼容处理，遇到新字段不兼容时在该函数中 `pop`。

### task 描述不匹配导致 VLA 抓取失败

**现象**：动作轨迹看似合理（有移动），但始终抓不到物体。ACT/Diffusion 无此问题。

**原因**：VLA 策略以 task 文本做条件推理。训练用 `"put the cube in the basket"`，推理用 `"pick cube"` → 条件偏移。

**解决**：确保转换时 `--task` 和推理时 `--task` **完全一致**（包括大小写和冠词）。
