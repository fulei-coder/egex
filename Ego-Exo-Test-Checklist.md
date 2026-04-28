# Ego-Exo 测试顺序与验收清单

这份文档按推荐顺序列出你现在需要做的测试。

目标不是一次把所有东西跑完，而是按依赖关系逐步确认：

1. 环境能不能导入
2. 采集出的 HDF5 对不对
3. 转换出的 LeRobot 数据集对不对
4. 几何投影能不能跑
5. 推理链路能不能在线算 grounding
6. `egexo_smolvla` 训练入口能不能启动
7. 消融配置能不能批量生成和运行

---

## 0. 测试前准备

建议先进入你平时训练/推理使用的环境，再执行下面命令。

```bash
cd /home/tony/lerobot-realman-vla
```

如果你有 conda 环境，例如：

```bash
conda activate lerobot
```

建议先确认这些依赖在你的环境里存在：

```bash
python3 -c "import torch, h5py, yaml, cv2, numpy; print('basic deps ok')"
python3 -c "import lerobot; print('lerobot ok')"
```

预期效果：

- 第一条打印 `basic deps ok`
- 第二条打印 `lerobot ok`

如果失败：

- 缺 `torch`：训练/推理环境还没进对
- 缺 `h5py`：HDF5 检查和转换脚本还不能跑
- 缺 `lerobot`：训练入口和数据集转换还不能跑

---

## 1. 代码静态检查

先确认主要脚本至少没有 Python 语法错误。

```bash
python3 -m py_compile \
  scripts/collect_data.py \
  scripts/collect_data_uarm.py \
  scripts/convert_to_lerobot.py \
  scripts/check_egexo_hdf5.py \
  scripts/debug_project_grounding.py \
  scripts/inference.py \
  scripts/evaluate_replay.py
```

预期效果：

- 没有输出，命令直接返回

如果失败：

- 先修语法错误，再做后续测试

---

## 2. 采集链路测试

这一项需要机械臂、顶部相机、腕部相机、ROS2 相机话题都正常。

### 2.1 U-arm 采集入口

```bash
python3 scripts/collect_data_uarm.py \
  --arm-ip 192.168.1.18 \
  --save-dir data/raw_hdf5 \
  --task-name pick_cube \
  --fps 15 \
  --calibration-config configs/calibration_realman.yaml
```

推荐录制流程：

1. `c`
2. `w`
3. `s`
4. 做一小段示教
5. `d`
6. `q`

预期效果：

- 能看到顶部相机和腕部相机预览
- 终端打印顶部相机和腕部相机初始化成功
- 开始录制后，最终在 `data/raw_hdf5/` 下生成一个 `.hdf5`
- 保存时终端会打印：
  - `qpos shape=(N, 7)`
  - `action shape=(N, 7)`
  - `depth shape=(N, H, W)`
  - `ee_pose shape=(N, 6)`

如果失败：

- 顶部相机失败：先检查 RealSense 序列号
- 腕部相机无帧：先检查 ROS2 相机节点和话题
- 机械臂连接失败：先检查 `arm-ip`

---

## 3. HDF5 自检

录完一条之后，先做 HDF5 检查。

```bash
python3 scripts/check_egexo_hdf5.py --input data/raw_hdf5/pick_cube_0.hdf5
```

预期效果：

- 这些键应为 `OK`
  - `observations/qpos`
  - `action`
  - `observations/images/cam_high`
  - `observations/images/cam_wrist`
  - `observations/depth/cam_high`
  - `observations/ee_pose`
  - `metadata/cameras/cam_high/intrinsics`
  - `metadata/cameras/cam_high/T_base_cam`
  - `metadata/cameras/cam_wrist/intrinsics`
  - `metadata/cameras/cam_wrist/T_ee_cam`
- `qpos/action 7D: True`
- `frames consistent: True`
- `depth all zero: False`

可以接受但要注意的情况：

- `wrist dark ratio` 偏高：说明腕部图像太暗
- `cam_wrist` 标定键存在，但当前值可能还是占位值

---

## 4. 原始数据离线回放检查

不依赖机械臂，只检查 raw HDF5 的统计是否合理。

```bash
python3 scripts/evaluate_replay.py --input data/raw_hdf5
```

预期效果：

- 输出 JSON
- `mode` 是 `raw_hdf5`
- `action_dim_ok` 是 `true`
- `grounding_valid_rate` 至少能输出一个数值
- `phase_distribution` 至少能输出 transport/contact 比例

重点看：

- 如果 `action_dim_ok` 不是 `true`，说明 7D 接口被破坏了
- 如果 `grounding_valid_rate` 一直是 `0.0`，说明当前 grounding 还没真正有效

---

## 5. LeRobot 数据集转换

### 5.1 旧格式转换

先确认旧链路没坏。

```bash
python3 scripts/convert_to_lerobot.py \
  --input-dir data/raw_hdf5 \
  --output-dir data/pick_cube_15fps \
  --repo-id local/pick_cube_15fps \
  --fps 15 \
  --task "pick up the cube"
```

预期效果：

- 正常创建 `data/pick_cube_15fps`
- 能完成 `dataset.consolidate()`

### 5.2 Ego-Exo 格式转换

```bash
python3 scripts/convert_to_lerobot.py \
  --input-dir data/raw_hdf5 \
  --output-dir data/pick_cube_egexo_15fps \
  --repo-id local/pick_cube_egexo_15fps \
  --fps 15 \
  --task "pick up the cube" \
  --enable-egexo \
  --calib configs/calibration_realman.yaml \
  --runtime-config configs/egexo_runtime.yaml
```

预期效果：

- 正常创建 `data/pick_cube_egexo_15fps`
- 不因为单帧 `grounding invalid` 崩溃
- 最终转换成功

---

## 6. 转换后数据集离线检查

```bash
python3 scripts/evaluate_replay.py --input data/pick_cube_egexo_15fps
```

预期效果：

- 输出 JSON
- `mode` 是 `lerobot_dataset`
- `action_dim_ok` 是 `true`
- `grounding_feature_present` 是 `true`
- `phase_feature_present` 是 `true`

如果环境里装了 `pyarrow`，还应能看到：

- `grounding_valid_rate`
- `phase_distribution`

---

## 7. 几何投影调试

这一步验证 exo ROI 到 ego ROI 的投影逻辑。

```bash
python3 scripts/debug_project_grounding.py \
  --input data/raw_hdf5/pick_cube_0.hdf5 \
  --calib configs/calibration_realman.yaml \
  --runtime-config configs/egexo_runtime.yaml \
  --output debug/grounding
```

预期效果：

- 生成：
  - `debug/grounding/frame_000_exo_roi.jpg`
  - `debug/grounding/frame_000_ego_projected_roi.jpg`
  - `debug/grounding/summary.csv`
- 终端打印：
  - `valid_rate=...`
  - `roi_out_of_view_rate=...`
  - `mean_roi_area=...`
  - `median_depth=...`

重点解释：

- 如果 `projection_reason` 里大量出现 `invalid_ego_intrinsics`
  - 说明腕部相机内参还没填真实值
- 如果大量出现 `invalid_depth`
  - 说明 depth ROI 全零或深度范围不合理
- 如果大量出现 `roi_out_of_view`
  - 说明外参或手眼关系不对

---

## 8. 推理链路测试

这一项需要机械臂、相机和模型 checkpoint。

### 8.1 普通推理先验证旧链路没坏

```bash
python3 scripts/inference.py \
  --model outputs/smolvla_realman/checkpoints/last/pretrained_model \
  --task "pick up the cube" \
  --freq 15
```

预期效果：

- 正常加载模型
- 正常初始化相机和机械臂
- 能进入循环，不报新的 Ego-Exo 相关错误

### 8.2 开启实时 grounding

```bash
python3 scripts/inference.py \
  --model outputs/smolvla_realman/checkpoints/last/pretrained_model \
  --task "pick up the cube" \
  --freq 15 \
  --enable-egexo-grounding \
  --debug-egexo \
  --calib configs/calibration_realman.yaml \
  --egexo-runtime-config configs/egexo_runtime.yaml
```

预期效果：

- 推理不中断
- 会生成：
  - `debug/egexo/step_000000_exo.jpg`
  - `debug/egexo/step_000000_ego.jpg`
  - `debug/egexo/phase_log.csv`
- 终端日志里每隔若干步会出现：
  - `G:0` 或 `G:1`
  - `P:0.0` 或 `P:1.0`

重点说明：

- 如果现在腕部相机标定还是占位值，`G` 可能长期是 `0`
- 这不一定表示推理链路有 bug，可能只是 wrist 标定没补齐

---

## 9. `egexo_smolvla` 训练入口测试

这一步先看能不能正常进入训练框架。

```bash
bash scripts/train.sh egexo_smolvla
```

预期效果：

- 终端先打印 `PYTHONPATH`
- 能打印 Ego-Exo 训练适配器生成的临时配置路径
- 能进入 `lerobot.scripts.lerobot_train`
- 理想情况：至少进入 dataloader 初始化或第一个 batch

如果失败：

- 缺 `lerobot`：训练环境没装好
- 缺 `torch`：训练环境没装好
- 数据集路径不对：先检查 `configs/egexo_smolvla_realman.yaml`

---

## 10. 消融配置生成测试

先只生成，不运行。

```bash
bash scripts/run_egexo_ablation.sh configs/egexo_smolvla_realman.yaml print
```

预期效果：

- 终端会打印一个临时 `manifest.yaml`
- 会列出这些 variant：
  - `cam_high_only`
  - `cam_wrist_only`
  - `naive_concat`
  - `asymmetric`
  - `grounding`
  - `phase_head`
  - `full`
  - `full_no_aug`

---

## 11. 单个消融训练测试

建议先跑一个最小变体，例如 `full`。

终端会在上一步打印出临时配置路径，例如 `/tmp/egexo_ablation.xxxx/full.yaml`。

然后执行：

```bash
python3 -m realman_vla.policies.egexo_smolvla.train --config /tmp/egexo_ablation.xxxx/full.yaml
```

预期效果：

- 能进入训练
- 输出目录是 `outputs/ablation/full`

---

## 12. 批量消融训练

如果单个变体没问题，再跑全部。

```bash
bash scripts/run_egexo_ablation.sh configs/egexo_smolvla_realman.yaml run
```

预期效果：

- 会逐个运行所有 ablation variant
- 输出目录分别落在 `outputs/ablation/*`

---

## 13. 推荐测试顺序总结

如果你时间有限，建议严格按下面顺序：

1. 环境导入检查
2. `py_compile`
3. `collect_data_uarm.py`
4. `check_egexo_hdf5.py`
5. `evaluate_replay.py --input data/raw_hdf5`
6. `convert_to_lerobot.py` 旧格式
7. `convert_to_lerobot.py --enable-egexo`
8. `evaluate_replay.py --input data/pick_cube_egexo_15fps`
9. `debug_project_grounding.py`
10. `inference.py` 普通模式
11. `inference.py --enable-egexo-grounding`
12. `bash scripts/train.sh egexo_smolvla`
13. `run_egexo_ablation.sh print`
14. 单个 ablation
15. 全量 ablation

---

## 14. 当前最可能遇到的问题

### 1. 腕部相机标定还是占位值

现状：

- `configs/calibration_realman.yaml` 里的 `cam_wrist` 很可能还不是最终真实标定

会导致：

- `debug_project_grounding.py` 的 `projection_reason` 经常异常
- `inference.py` 里的 `grounding.valid` 长期偏低

### 2. 环境缺包

常见缺失：

- `torch`
- `lerobot`
- `h5py`
- `pyarrow`

### 3. 数据集路径不一致

训练前重点确认：

- `configs/egexo_smolvla_realman.yaml` 的 `dataset.root`
- 实际转换输出目录

---

## 15. 测完后建议反馈给我什么

如果你执行测试后要我继续帮你排查，最有用的是这些信息：

1. 你执行的具体命令
2. 完整报错
3. 生成的 `.hdf5` 或数据集路径
4. `debug/grounding/summary.csv` 的前几行
5. `debug/egexo/phase_log.csv` 的前几行
6. `bash scripts/train.sh egexo_smolvla` 卡在第几步

如果你愿意，后面你跑完其中几步，把输出贴给我，我可以继续按这个顺序陪你排。 
