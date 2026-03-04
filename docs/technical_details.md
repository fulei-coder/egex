# 技术细节

---

## 1. 数据格式

### HDF5（采集原始格式）

```python
{
    'observations/qpos':             (N, 7)  float32   # [j1..j6, gripper]
    'observations/images/cam_high':  (N, 480, 640, 3) uint8
    'observations/images/cam_wrist': (N, 480, 640, 3) uint8
    'action':                        (N, 7)  float32   # action[t] = qpos[t+1]
    'timestamps':                    (N,)    float64
    attrs: { 'fps': 30, 'sim': False }
}
```

`action[t] = qpos[t+1]`：当前帧的"目标动作"就是下一帧的关节状态。不需要额外标注，数据自监督。

### LeRobot v3 格式

Parquet（表格数据）+ MP4（视频编码，比逐帧存图节省 ~10x 空间）。

`stats.json` 存每个特征的 μ 和 σ，训练时自动 z-score 归一化：

$$x_{norm} = \frac{x - \mu}{\sigma}$$

推理时 postprocessor 做反向还原：

$$x_{raw} = x_{norm} \times \sigma + \mu$$

由 `DataProcessorPipeline` 自动处理。推理时如果 postprocessor 未加载，输出的归一化值直接发给机械臂 → 关节角接近 0 → 不动。

---

## 2. 策略架构

### ACT（Action Chunking with Transformers）

用 CVAE + Transformer 一次预测多步动作（chunk）。在简单任务上效果最好——164 条数据 31K 步即达 90%+ 成功率。

```
图像 → ResNet18 → Transformer Encoder (4层)
                         ↓
               CVAE: z ~ N(μ,σ²) [训练] / z=0 [推理]
                         ↓
               Transformer Decoder → 50 步动作序列
```

**关键设计**：

- **Action Chunking**：`chunk_size=50`，一次预测 50 步。减少决策频率 → 减少误差传播
- **Temporal Ensemble**：多个 chunk 重叠部分加权平均，权重 $w_k = e^{-\alpha k}$（α=0.01），消除 chunk 边界跳变
- **CVAE**：训练时编码"动作意图"到 z，推理时用先验 z=0。KL 权重 β=10——太大导致 mode collapse，太小则 z=0 偏离训练分布

**损失函数**：

$$\mathcal{L} = \underbrace{||a - \hat{a}||_1}_{\text{L1}} + \beta \cdot \underbrace{D_{KL}(q(z|a,o) \| p(z))}_{\text{KL}}$$

### Diffusion Policy

将动作生成建模为去噪扩散：

```
a_T ~ N(0,I) → 条件 UNet 反复去噪 → a_0（动作）
```

训练学预测噪声 ε，推理从噪声迭代去噪。DDIM 可将 100 步缩减为 10 步，质量几乎无损。

核心优势：天然支持多模态分布。同一观测下左右两侧都能抓时，MSE 回归会预测中间（无效），Diffusion 可以采样到任一模式。

### VQ-BeT

两阶段设计：

1. **阶段 1（0~20K 步）**：VQ-VAE 学 Codebook，离散化连续动作
2. **阶段 2（20K+ 步）**：冻结 VQ-VAE，GPT 预测 code 序列

不能端到端训练——Codebook 需要先稳定，否则 GPT 梯度会干扰其收敛。

注意：VQ-BeT 只支持单路相机输入。

### SmolVLA / Pi0（VLA 策略）

利用预训练 VLM 做机器人控制：

```
图像 + 语言指令 + 关节角
    → VLM Backbone (SmolVLM-500M / PaliGemma-2B)
    → Action Expert (Flow Matching)
    → 动作序列
```

**Flow Matching vs Diffusion**：

| | Diffusion | Flow Matching |
|---|---|---|
| 正向过程 | 逐步加噪 | 直线插值 $x_t = (1-t)x_0 + t\epsilon$ |
| 训练目标 | 预测噪声 ε | 预测速度场 v |
| 推理效率 | 多步去噪 | 通常更少步 |

Flow Matching 把噪声到数据的变换看成"直线路径"，比 Diffusion 的"曲折路径"更高效。

**冻结策略**：VLA 参数量大（500M~2B），冻结视觉编码器 + 只训练 Action Expert 可将训练参数量降一个数量级。SmolVLA 默认开启。

---

## 3. 推理优化

### EMA 动作平滑

$$a_{smooth}^{(t)} = \alpha \cdot a_{pred}^{(t)} + (1 - \alpha) \cdot a_{smooth}^{(t-1)}$$

α=0.3 较平滑，α=0.7 响应快。ACT 有 Temporal Ensemble 通常不需要 EMA，Diffusion/VQ-BeT 建议开。

### 死区过滤

关节变化小于阈值时不发送指令：

$$\text{send} = \max_i |a_i^{(t)} - a_i^{(t-1)}| > \delta \quad (\delta = 0.5°)$$

所有策略都建议开，无负面影响。

### 异步夹爪控制

Modbus RTU 9600 波特率下单次读写 300-500ms。同步执行会将 30Hz 控制频率拖到 1-2Hz。

```python
# 异步写入 + 缓存读取
if gripper_state_changed:
    threading.Thread(target=write_gripper, daemon=True).start()
```

夹爪状态只有开/闭，不需要实时读取，用缓存替代。效果：1-2Hz → 30Hz+。

### Temporal Ensemble（ACT 专用）

多个 chunk 的重叠预测取加权平均：

```
时刻t:   chunk_1 = [a₁, a₂, a₃, ...]
时刻t+1: chunk_2 = [_,  a₂', a₃', ...]

执行 a₃ = Σ wₖ·a₃⁽ᵏ⁾ / Σ wₖ
```

权重指数衰减（coeff=0.01），越早的预测权重越大。

---

## 4. 设计决策

### 关节空间 vs 笛卡尔空间

| | 关节空间 | 笛卡尔空间 |
|---|---|---|
| 唯一性 | 唯一解 | 可能有多个 IK 解 |
| 奇异点 | 无 | 存在奇异位型 |
| 标签构造 | `action = next_qpos`，无需额外标注 | 需要正/逆运动学 |

关节空间下 `action = next_qpos` 天然自监督，且位置控制对频率偏差有容错（最终都会到达目标位置）。

### 频率三者对齐

```
采集 fps = 转换 fps = 推理 freq
```

BC 学的是 `a_t = s_{t+1}`（1/FPS 秒后的目标）。推理频率不同 = 改变时间尺度。30Hz 数据 + 15Hz 推理 → 速度减半。位置控制对此有一定容错，但速度控制下会累积位移误差。

---

## 5. 参数调整

### batch_size / num_workers 决策

```
显存没满 && GPU 利用率高        → 加 batch_size
GPU 利用率低 && data_s >> updt_s → IO 瓶颈，加 num_workers
Bus error                        → 减 batch_size 或 num_workers
```

**实测可用配置**：

| | ACT | SmolVLA | Pi0 |
|---|---|---|---|
| 4080 12GB | batch=64, workers=2 | batch=4, workers=1 | ❌ |
| A800 80GB | batch=128, workers=8 | batch=4, workers=4 | batch=2, workers=4 + gradient_ckpt |

### 学习率

LeRobot VLA 策略有隐藏机制：`use_policy_training_preset=True` 时，`validate()` 会**覆盖 yaml 中配置的 optimizer 参数**（包括 lr）。

解决：设 `use_policy_training_preset: false`，或修改 `configuration_pi0.py` 里 preset 默认值。

### n_obs_steps

观测历史帧数。不是越多越好：

- SmolVLA: obs=1 → obs=7 后 loss 高 10 倍（VLM 更适合单帧 rich feature 而非多帧序列）
- ACT: obs_steps 和 chunk_size 有耦合，随意改可能导致维度不匹配

### freeze_vision_encoder

VLA 的视觉编码器已在大规模数据上预训练，微调时冻结可以：
- 减少可训练参数（~10x）
- 防止小数据集过拟合
- 减少显存

SmolVLA 默认开启，Pi0 默认关闭。数据量 <100 条时建议所有 VLA 都开。
