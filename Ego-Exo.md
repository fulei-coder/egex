请基于当前仓库 `fulei-coder/ros2-realman-uarm-vla` 修改代码，目标是在不破坏现有 ACT / Diffusion / SmolVLA / Pi0 基线的前提下，实现一个可训练、可推理、可消融的 **Ego-Exo Grounded SmolVLA** 原型。

当前仓库不是纯模型仓库，而是完整链路仓库：

```text
scripts/collect_data.py          # Vive / 示教采集，双相机 HDF5
scripts/collect_data_uarm.py     # U-arm 遥操作采集，7D qpos/action
scripts/convert_to_lerobot.py    # HDF5 → LeRobot v3
scripts/train.sh                 # 根据 configs/*_realman.yaml 启动训练
scripts/inference.py             # 多策略实机推理
hardware/uarm_realman_mapper.py
hardware/realman_teleop_controller.py
configs/smolvla_realman.yaml
configs/uarm_realman_map.yaml
```

请优先在这些现有入口上扩展，不要重写整套工程。

---

# 1. 总体目标

新增一个策略路线：

```text
egexo_smolvla
```

核心思想：

```text
cam_high / D455 / 顶部环境视角 = exo view，负责全局定位、粗几何、遮挡判断
cam_wrist / DS87 / 腕部视角 = ego view，负责接触前局部细节、把手、孔位、边缘、夹爪相对关系
```

不要把两个相机继续当成完全同质的多图像输入。要显式区分：

```text
observation.images.cam_high   → exo image
observation.images.cam_wrist  → ego image
```

第一版不要做 critic、主动视角搜索、复杂 MoE、dense reconstruction。先实现：

```text
1. 数据层增加 D455 depth、ee_pose、标定参数、grounding/phase 标签
2. 几何模块：exo RGB-D ROI → 3D → ego wrist ROI
3. 模型层：EgExoSmolVLA，基于 SmolVLA 扩展
4. 推理层：实时读取 D455 RGB-D + DS87 RGB，计算 grounding，送入策略
5. 消融开关：能关闭 exo、ego、grounding、phase head
```

---

# 2. 不能破坏的现有接口

必须保留现有 HDF5 键：

```text
observations/qpos
observations/images/cam_high
observations/images/cam_wrist
action
timestamps
```

必须保留现有 LeRobot 键：

```text
observation.state
observation.images.cam_high
observation.images.cam_wrist
action
task
```

必须保留 7D 状态和动作约定：

```text
qpos:   [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
action: [joint_1, joint_2, joint_3, joint_4, joint_5, joint_6, gripper]
```

当前夹爪未启用时，第 7 维继续作为 placeholder。不要把状态改回 6 维。

所有新增字段都用“附加字段”的方式加入，不能删旧字段。

---

# 3. 新增配置文件

新增：

```text
configs/egexo_smolvla_realman.yaml
configs/calibration_realman.yaml
configs/egexo_runtime.yaml
```

注意：`train.sh` 当前按 `${POLICY}_realman.yaml` 找配置，所以策略名用：

```bash
bash scripts/train.sh egexo_smolvla
```

对应配置文件必须叫：

```text
configs/egexo_smolvla_realman.yaml
```

不要叫 `smolvla_egexo_realman.yaml`，否则现有 `train.sh` 找不到。

---

## 3.1 `configs/egexo_smolvla_realman.yaml`

建议内容：

```yaml
dataset:
  repo_id: local/pick_cube_egexo_15fps
  root: data/pick_cube_egexo_15fps
  video_backend: pyav

policy:
  type: egexo_smolvla
  repo_id: local/egexo_smolvla_realman

  base_policy_type: smolvla
  vlm_model_name: HuggingFaceTB/SmolVLM2-500M-Video-Instruct
  load_vlm_weights: true

  n_obs_steps: 1
  chunk_size: 50
  n_action_steps: 50
  resize_imgs_with_padding: [512, 512]
  tokenizer_max_length: 48

  freeze_vision_encoder: true
  train_expert_only: true
  train_state_proj: true

  num_vlm_layers: 16
  num_expert_layers: -1
  expert_width_multiplier: 0.75
  attention_mode: cross_attn
  num_steps: 10

  image_keys:
    exo: observation.images.cam_high
    ego: observation.images.cam_wrist

  state_key: observation.state
  ee_pose_key: observation.ee_pose
  ego_roi_key: observation.grounding.ego_roi
  grounding_valid_key: observation.grounding.valid
  phase_key: observation.phase

  egexo:
    use_exo: true
    use_ego: true
    use_view_embedding: true
    use_crossview_grounding: true
    grounding_mode: soft_mask
    use_grounding_loss: true
    use_phase_head: true
    phase_mode: dual_head
    action_mixing: soft

  loss:
    action_weight: 1.0
    grounding_weight: 0.1
    phase_weight: 0.05
    cross_view_consistency_weight: 0.0

training:
  batch_size: 4
  steps: 50000
  save_checkpoint: true
  save_freq: 2500
  eval_freq: 5000
  log_freq: 50
  output_dir: outputs/egexo_smolvla_realman
  seed: 42
  num_workers: 2

wandb:
  enable: false
```

字段名可根据 LeRobot 版本要求调整，但语义必须保留。

---

## 3.2 `configs/calibration_realman.yaml`

新增标定配置，不要把标定矩阵硬编码在 Python 文件里。

```yaml
cameras:
  cam_high:
    role: exo
    type: realsense
    serial: "108222250854"
    width: 640
    height: 480
    fps: 15
    intrinsics:
      fx: 0.0
      fy: 0.0
      cx: 0.0
      cy: 0.0
    T_base_cam:
      data: [1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1]

  cam_wrist:
    role: ego
    type: ds87_ros
    rgb_topic: "/Scepter/color/image_raw"
    width: 640
    height: 480
    intrinsics:
      fx: 0.0
      fy: 0.0
      cx: 0.0
      cy: 0.0
    T_ee_cam:
      data: [1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1, 0,
             0, 0, 0, 1]

geometry:
  min_depth_m: 0.15
  max_depth_m: 2.0
  roi_expand_ratio: 1.8
  reprojection_soft_sigma_px: 12.0
  max_reprojection_error_px: 40.0
```

实际数值后续用标定结果填充。第一版允许默认 identity，但必须打印 warning，不能静默使用假标定。

---

## 3.3 `configs/egexo_runtime.yaml`

用于推理时控制目标粗定位、grounding、debug 输出。

```yaml
target_locator:
  mode: manual_roi   # manual_roi / color_heuristic / external_detector
  manual_roi_xyxy: [220, 160, 420, 360]
  target_color: null

runtime_grounding:
  enabled: true
  fallback_when_invalid: true
  use_soft_mask: true

phase:
  enabled: true
  distance_threshold_m: 0.08
  hard_switch_in_inference: false

debug:
  enabled: true
  save_dir: debug/egexo
  save_every_n_steps: 10
  draw_exo_roi: true
  draw_ego_projected_roi: true
  save_phase_prob: true
```

---

# 4. 修改采集层

需要同时修改：

```text
scripts/collect_data.py
scripts/collect_data_uarm.py
```

当前两个采集脚本都已经保存双相机 RGB。下一步要补充：

```text
D455 depth
ee_pose
相机内参
相机外参
相机时间戳
可选 target_roi_exo
```

---

## 4.1 修改 D455 相机类

当前 `D435Camera` / `RealSenseCamera` 只稳定使用 color。请改成 RGB-D 采集类，但保留原类名或兼容包装，避免旧代码大面积改动。

建议改成：

```python
class D435Camera:
    def __init__(self, serial_number, width=640, height=480, fps=15, enable_depth=False):
        ...
        self.enable_depth = enable_depth
        self.latest_color = np.zeros((height, width, 3), dtype=np.uint8)
        self.latest_depth = np.zeros((height, width), dtype=np.uint16)
        self.intrinsics = None
```

启动 RealSense 时：

```python
self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

if enable_depth:
    self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
    self.align = rs.align(rs.stream.color)
```

更新循环：

```python
frames = self.pipeline.wait_for_frames(timeout_ms=2000)

if self.enable_depth:
    frames = self.align.process(frames)

color_frame = frames.get_color_frame()
depth_frame = frames.get_depth_frame() if self.enable_depth else None
```

新增接口：

```python
def get_frame(self):
    return latest_color.copy()

def get_depth_frame(self):
    return latest_depth.copy()

def get_intrinsics(self):
    return {
        "fx": ...,
        "fy": ...,
        "cx": ...,
        "cy": ...,
        "width": self.width,
        "height": self.height,
        "depth_scale": self.depth_scale,
    }
```

注意：旧代码调用 `get_frame()` 的地方不能坏。

---

## 4.2 修改 `DataRecorder`

当前 `collect_data_uarm.py` 里的 `DataRecorder` 已经比 `collect_data.py` 更完整，包含：

```text
qpos
action
images_top
images_wrist
timestamps
leader timestamp
camera age
camera mean
movej_ret_code
```

请优先以 `collect_data_uarm.py` 的 recorder 为主线扩展，然后把 `collect_data.py` 也对齐。

新增 buffer 字段：

```python
"depth_top": [],
"ee_pose": [],
"target_roi_exo": [],
"target_3d_base": [],
```

每一帧录制时：

```python
img_top = self.cam_top.get_frame()
depth_top = self.cam_top.get_depth_frame()
img_wrist = self.cam_wrist.get_frame()

code, state = arm.rm_get_current_arm_state()
qpos = state["joint"][:6]
ee_pose = state.get("pose", None)
```

如果 SDK 返回字段名不是 `pose`，请根据 RealMan SDK 实际返回结构适配。不要猜错后直接写死。建议写一个兼容函数：

```python
def extract_ee_pose_from_realman_state(state):
    if "pose" in state:
        return np.asarray(state["pose"], dtype=np.float32)
    if "tool_pose" in state:
        return np.asarray(state["tool_pose"], dtype=np.float32)
    return np.zeros(6, dtype=np.float32)
```

保存 HDF5 时新增：

```python
f.create_dataset("observations/depth/cam_high", data=depth_top, compression="gzip")
f.create_dataset("observations/ee_pose", data=ee_pose.astype(np.float32))
```

同时保存标定元数据：

```python
f.create_dataset("metadata/cameras/cam_high/intrinsics", data=np.asarray([...], dtype=np.float32))
f.create_dataset("metadata/cameras/cam_wrist/intrinsics", data=np.asarray([...], dtype=np.float32))
f.create_dataset("metadata/cameras/cam_high/T_base_cam", data=np.asarray(..., dtype=np.float32))
f.create_dataset("metadata/cameras/cam_wrist/T_ee_cam", data=np.asarray(..., dtype=np.float32))
```

不要删除旧字段：

```text
observations/images/cam_high
observations/images/cam_wrist
```

---

# 5. 新增几何模块

新增目录：

```text
src/realman_vla/
src/realman_vla/geometry/
src/realman_vla/vision/
src/realman_vla/policies/
```

新增文件：

```text
src/realman_vla/geometry/crossview.py
src/realman_vla/geometry/calibration.py
src/realman_vla/vision/target_locator.py
```

---

## 5.1 `crossview.py`

实现以下函数：

```python
def depth_roi_to_3d(
    depth: np.ndarray,
    roi_xyxy: np.ndarray,
    intrinsics: dict,
    min_depth_m: float,
    max_depth_m: float,
) -> tuple[np.ndarray, bool]:
    """
    输入 exo ROI + depth，输出目标在 exo camera frame 下的 3D centroid。
    第一版使用 ROI 内 depth median，不要做 dense point cloud。
    """
```

```python
def transform_point(T_dst_src: np.ndarray, p_src: np.ndarray) -> np.ndarray:
    """
    4x4 齐次变换。
    """
```

```python
def project_point_to_image(
    p_cam: np.ndarray,
    intrinsics: dict,
) -> tuple[np.ndarray, bool]:
    """
    3D point in camera frame → pixel uv。
    """
```

```python
def project_exo_roi_to_ego(
    exo_roi_xyxy,
    exo_depth,
    exo_intrinsics,
    ego_intrinsics,
    T_base_exo,
    T_base_ee,
    T_ee_ego,
    image_size,
    cfg,
):
    """
    cam_high ROI + depth
    → target_3d_exo
    → target_3d_base
    → target_3d_ego
    → cam_wrist projected ROI
    """
```

返回：

```python
{
    "target_3d_base": np.ndarray shape (3,),
    "ego_roi_xyxy": np.ndarray shape (4,),
    "valid": bool,
    "reason": str,
}
```

---

## 5.2 `target_locator.py`

第一版不要接复杂 detector。先做可控的三种模式：

```python
class TargetLocator:
    def __init__(self, cfg):
        self.mode = cfg["target_locator"]["mode"]

    def locate(self, image_bgr):
        if self.mode == "manual_roi":
            return np.array(cfg["manual_roi_xyxy"], dtype=np.float32), True
        elif self.mode == "color_heuristic":
            return locate_by_color(...)
        elif self.mode == "external_detector":
            raise NotImplementedError
```

第一版必须支持 `manual_roi`，保证闭环能跑。

---

# 6. 修改 `convert_to_lerobot.py`

当前转换脚本只读：

```text
qpos
action
cam_high
cam_wrist
```

请增加 `--enable-egexo` 参数。

默认行为保持不变：

```bash
python scripts/convert_to_lerobot.py ...
```

仍然生成旧格式数据集。

启用新格式：

```bash
python scripts/convert_to_lerobot.py \
  --input-dir data/raw_hdf5/pick_cube \
  --output-dir data/pick_cube_egexo_15fps \
  --repo-id local/pick_cube_egexo_15fps \
  --fps 15 \
  --task "pick up the cube" \
  --enable-egexo \
  --calib configs/calibration_realman.yaml \
  --runtime-config configs/egexo_runtime.yaml
```

启用 `--enable-egexo` 后，LeRobot features 增加：

```python
"observation.ee_pose": {
    "dtype": "float32",
    "shape": (6,),
    "names": ["x", "y", "z", "rx", "ry", "rz"],
},

"observation.grounding.ego_roi": {
    "dtype": "float32",
    "shape": (4,),
    "names": ["x1", "y1", "x2", "y2"],
},

"observation.grounding.valid": {
    "dtype": "float32",
    "shape": (1,),
    "names": ["valid"],
},

"observation.target_3d_base": {
    "dtype": "float32",
    "shape": (3,),
    "names": ["x", "y", "z"],
},

"observation.phase": {
    "dtype": "float32",
    "shape": (1,),
    "names": ["phase"],
},
```

不要把 `cam_high` / `cam_wrist` 重命名为 `exo` / `ego`，否则现有推理和旧模型会断。只在配置里解释：

```yaml
image_keys:
  exo: observation.images.cam_high
  ego: observation.images.cam_wrist
```

---

## 6.1 phase 标签生成规则

第一版半自动生成：

```python
distance = np.linalg.norm(ee_position - target_3d_base)

if distance > distance_threshold_m:
    phase = 0.0  # transport / approach
else:
    phase = 1.0  # contact / refine
```

如果 `target_3d_base` invalid：

```python
phase = 0.0
grounding.valid = 0.0
ego_roi = [0, 0, 0, 0]
```

不要因为单帧 grounding invalid 就中断整个 episode 转换。

---

# 7. 新增自检脚本

新增：

```text
scripts/check_egexo_hdf5.py
scripts/debug_project_grounding.py
```

---

## 7.1 `check_egexo_hdf5.py`

检查：

```text
observations/qpos 是否 7D
action 是否 7D
observations/images/cam_high 是否存在
observations/images/cam_wrist 是否存在
observations/depth/cam_high 是否存在
observations/ee_pose 是否存在
metadata/cameras/... 是否存在
帧数是否全部一致
depth 是否全零
wrist 是否暗帧过多
```

运行：

```bash
python scripts/check_egexo_hdf5.py --input data/raw_hdf5/pick_cube/task_pick_cube_0.hdf5
```

---

## 7.2 `debug_project_grounding.py`

输入一个 HDF5 episode，输出可视化：

```text
debug/grounding/frame_000_exo_roi.jpg
debug/grounding/frame_000_ego_projected_roi.jpg
debug/grounding/summary.csv
```

必须统计：

```text
valid_rate
roi_out_of_view_rate
mean_roi_area
median_depth
```

这是验证标定和投影逻辑的第一优先级工具。

---

# 8. 模型层实现路线

当前仓库本身没有明显的自定义 `models/` 训练包，训练是直接调用：

```bash
python -m lerobot.scripts.lerobot_train --config configs/xxx_realman.yaml
```

所以不能只在根目录随便加 `models/egexo_smolvla.py`，否则 LeRobot trainer 不会自动识别。

请按下面方式做。

---

## 8.1 新增本地 Python package

新增：

```text
src/realman_vla/__init__.py
src/realman_vla/policies/__init__.py
src/realman_vla/policies/egexo_smolvla/
src/realman_vla/policies/egexo_smolvla/__init__.py
src/realman_vla/policies/egexo_smolvla/configuration_egexo_smolvla.py
src/realman_vla/policies/egexo_smolvla/modeling_egexo_smolvla.py
```

同时修改 `scripts/train.sh`，在运行训练前加入：

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

推理脚本 `scripts/inference.py` 也要在顶部加入：

```python
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
```

---

## 8.2 LeRobot policy 注册

需要根据当前安装的 LeRobot 版本确认 policy registry 机制。

目标是让配置：

```yaml
policy:
  type: egexo_smolvla
```

能被：

```bash
python -m lerobot.scripts.lerobot_train --config configs/egexo_smolvla_realman.yaml
```

正确识别。

如果 LeRobot 要求策略放在 `lerobot.policies.*` 命名空间，则采用兼容方式：

```text
src/lerobot/policies/egexo_smolvla/
```

并在其中实现：

```text
configuration_egexo_smolvla.py
modeling_egexo_smolvla.py
```

如果 LeRobot 支持外部注册，则用外部注册，避免复制 `lerobot` 包。

验收标准：

```bash
python -c "from lerobot.policies.egexo_smolvla.modeling_egexo_smolvla import EgExoSmolVLAPolicy"
```

和：

```bash
bash scripts/train.sh egexo_smolvla
```

至少能进入 dataloader 初始化阶段。

---

## 8.3 EgExoSmolVLA 模型设计

第一版不要重写 SmolVLA 主干。基于现有 SmolVLA 做 wrapper。

模型输入：

```python
observation.images.cam_high
observation.images.cam_wrist
observation.state
observation.ee_pose
observation.grounding.ego_roi
observation.grounding.valid
observation.phase
task
```

内部逻辑：

```text
1. cam_high → exo branch
2. cam_wrist → ego branch
3. ego_roi → wrist soft mask / crop prior
4. view embedding 区分 exo / ego
5. phase gate 输出 transport/contact 概率
6. transport action head 和 contact action head 输出动作
7. 根据 phase_prob 混合动作
```

第一版可分三阶段实现：

### 阶段 A：最小可训练版

```text
仍使用 SmolVLA 单 action expert
但对 cam_wrist 做 projected ROI soft mask 或 ROI crop
加入 view embedding
加入 grounding/phase 辅助 loss 的接口
```

这个阶段必须先跑通训练。

### 阶段 B：phase embedding 版

```text
把 phase 作为额外 token / embedding 输入 action expert
暂时不拆双 action head
```

这个阶段实现成本较低。

### 阶段 C：dual action head 版

```text
transport_action_head
contact_action_head
phase_gate
action = p_transport * action_transport + p_contact * action_contact
```

论文最终建议用阶段 C，但代码第一版允许先完成 A/B。

---

# 9. 修改推理脚本 `scripts/inference.py`

当前推理主循环结构是：

```text
get qpos
get cam_high
get cam_wrist
add task
preprocessor
policy.select_action
postprocessor
robot.set_qpos
```

请在 `preprocessor` 之前插入 egexo grounding 逻辑。

---

## 9.1 新增命令行参数

在 parser 里新增：

```python
parser.add_argument("--egexo-runtime-config", type=str, default="configs/egexo_runtime.yaml")
parser.add_argument("--calib", type=str, default="configs/calibration_realman.yaml")
parser.add_argument("--enable-egexo-grounding", action="store_true")
parser.add_argument("--disable-egexo-grounding", action="store_true")
parser.add_argument("--debug-egexo", action="store_true")
```

逻辑：

```python
enable_egexo = (
    policy_type == "egexo_smolvla"
    or args.enable_egexo_grounding
) and not args.disable_egexo_grounding
```

---

## 9.2 修改 policy loader

在 `load_policy()` 中增加：

```python
elif policy_type == "egexo_smolvla":
    from realman_vla.policies.egexo_smolvla.modeling_egexo_smolvla import EgExoSmolVLAPolicy
    policy = EgExoSmolVLAPolicy.from_pretrained(str(model_path))
```

如果最终使用 `lerobot.policies.egexo_smolvla` 命名空间，则 import 路径按实际实现调整。

---

## 9.3 RealSenseCamera 支持 depth

当前 `inference.py` 内的 `RealSenseCamera` 只启用 color。请改成支持：

```python
RealSenseCamera(serial_number, width=640, height=480, fps=15, enable_depth=False)
```

当模型 input features 需要 grounding，或者 `enable_egexo=True` 时：

```python
cam_top = RealSenseCamera(args.cam_top, fps=args.freq, enable_depth=True)
```

推理循环中获取：

```python
img_top_bgr = cam_top.get_frame()
depth_top = cam_top.get_depth_frame()
img_wrist_bgr = cam_wrist.get_frame()
```

---

## 9.4 RobotController 增加 ee_pose

在 `RobotController` 里新增：

```python
def get_ee_pose(self):
    with self.lock:
        code, state = self.arm.rm_get_current_arm_state()
        if code != 0:
            return np.zeros(6, dtype=np.float32)
        return extract_ee_pose_from_realman_state(state)
```

不要在推理循环里重复写 SDK 解析逻辑。

---

## 9.5 推理 observation 增加 grounding 字段

在主循环中：

```python
observation = {
    "observation.state": torch.from_numpy(qpos).float(),
}
```

后面继续保留：

```python
observation["observation.images.cam_high"] = ...
observation["observation.images.cam_wrist"] = ...
```

新增：

```python
if enable_egexo:
    ee_pose = robot.get_ee_pose()
    exo_roi, roi_ok = target_locator.locate(img_top_bgr)

    grounding = project_exo_roi_to_ego(
        exo_roi_xyxy=exo_roi,
        exo_depth=depth_top,
        exo_intrinsics=calib.cam_high.intrinsics,
        ego_intrinsics=calib.cam_wrist.intrinsics,
        T_base_exo=calib.cam_high.T_base_cam,
        T_base_ee=ee_pose_to_T_base_ee(ee_pose),
        T_ee_ego=calib.cam_wrist.T_ee_cam,
        image_size=(640, 480),
        cfg=runtime_cfg,
    )

    observation["observation.ee_pose"] = torch.from_numpy(ee_pose).float()
    observation["observation.grounding.ego_roi"] = torch.from_numpy(grounding["ego_roi_xyxy"]).float()
    observation["observation.grounding.valid"] = torch.tensor([float(grounding["valid"])], dtype=torch.float32)
    observation["observation.target_3d_base"] = torch.from_numpy(grounding["target_3d_base"]).float()

    phase = estimate_phase(ee_pose, grounding["target_3d_base"], grounding["valid"], runtime_cfg)
    observation["observation.phase"] = torch.tensor([phase], dtype=torch.float32)
```

如果 grounding invalid：

```python
observation["observation.grounding.valid"] = tensor([0.0])
observation["observation.grounding.ego_roi"] = tensor([0, 0, 0, 0])
```

不能让推理崩溃。

---

# 10. Debug 可视化

推理时必须能保存：

```text
debug/egexo/step_000010_exo.jpg
debug/egexo/step_000010_ego.jpg
debug/egexo/phase_log.csv
```

exo 图上画：

```text
target_roi_exo
target_3d depth value
```

ego 图上画：

```text
projected ego ROI
grounding valid / invalid
phase probability
```

`phase_log.csv` 字段：

```text
step,time,grounding_valid,phase_transport,phase_contact,target_x,target_y,target_z,ego_roi_x1,ego_roi_y1,ego_roi_x2,ego_roi_y2
```

这部分非常重要，因为本方法最大风险是标定误差和投影误差。没有 debug 图，无法判断失败来自模型还是几何。

---

# 11. 消融开关

`configs/egexo_smolvla_realman.yaml` 和模型都必须支持：

```yaml
ablation:
  use_exo: true
  use_ego: true
  use_depth_exo: true
  use_crossview_grounding: true
  use_grounding_loss: true
  use_phase_head: true
  use_dual_action_head: true
  use_view_embedding: true
```

至少能跑以下配置：

```text
1. cam_high only
2. cam_wrist only
3. cam_high + cam_wrist naive concat
4. asymmetric dual-view
5. asymmetric + geometry grounding
6. asymmetric + geometry grounding + phase head
7. full model + / - data augmentation
```

重点对照不是 ACT，而是：

```text
SmolVLA + naive dual-view concat
vs
EgExoSmolVLA
```

---

# 12. 新增实验脚本

新增：

```text
scripts/run_egexo_ablation.sh
scripts/evaluate_replay.py
```

`run_egexo_ablation.sh` 负责自动改配置或传 override，输出目录类似：

```text
outputs/ablation/cam_high_only
outputs/ablation/cam_wrist_only
outputs/ablation/naive_concat
outputs/ablation/asymmetric
outputs/ablation/grounding
outputs/ablation/full
```

`evaluate_replay.py` 用离线 dataset 做 sanity check：

```text
1. 能否读取 egexo 数据集
2. policy.select_action 是否输出 7D action
3. grounding valid rate 是否合理
4. phase 标签分布是否合理
5. action shape 是否和现有 inference.py 一致
```

---

# 13. 修改 `scripts/train.sh`

当前 `train.sh` 已经按策略名找配置。只需要做兼容增强。

增加：

```bash
export PYTHONPATH="$(pwd)/src:${PYTHONPATH}"
```

增加打印：

```bash
echo "PYTHONPATH: ${PYTHONPATH}"
```

保留原逻辑：

```bash
CONFIG_FILE="${CONFIG_DIR}/${POLICY}_realman.yaml"
CMD="python -m lerobot.scripts.lerobot_train --config ${CONFIG_FILE}"
```

如果 LeRobot 当前版本实际要求 `--config_path` 而不是 `--config`，请统一修正，但不能影响旧策略。

---

# 14. 修改 `scripts/view_hdf5.py`

让它能显示新字段：

```text
depth/cam_high
ee_pose
grounding ego_roi
phase
```

显示方式：

```text
左：cam_high RGB + exo ROI
中：depth colormap
右：cam_wrist RGB + projected ego ROI
底部文字：qpos/action/ee_pose/phase/valid
```

---

# 15. 实施顺序

请严格按下面顺序做，不要一上来就改模型大结构。

---

## Milestone 1：数据不破坏扩展

完成：

```text
collect_data.py / collect_data_uarm.py 增加 depth、ee_pose、calibration metadata
convert_to_lerobot.py 增加 --enable-egexo，但默认旧行为不变
check_egexo_hdf5.py
```

验收：

```bash
python scripts/collect_data_uarm.py --arm-ip 192.168.1.18 --save-dir data/raw_hdf5 --task-name pick_cube --fps 15
python scripts/check_egexo_hdf5.py --input data/raw_hdf5/pick_cube_0.hdf5
python scripts/convert_to_lerobot.py ...                 # 旧格式仍可跑
python scripts/convert_to_lerobot.py ... --enable-egexo  # 新格式可跑
```

---

## Milestone 2：几何投影先跑通

完成：

```text
crossview.py
calibration.py
target_locator.py
debug_project_grounding.py
```

验收：

```bash
python scripts/debug_project_grounding.py \
  --input data/raw_hdf5/pick_cube_0.hdf5 \
  --calib configs/calibration_realman.yaml \
  --runtime-config configs/egexo_runtime.yaml \
  --output debug/grounding
```

必须输出：

```text
exo ROI 图
ego projected ROI 图
summary.csv
```

在这个阶段先不要训练模型。

---

## Milestone 3：推理链路接入 grounding，但暂不换模型

完成：

```text
inference.py 支持 --enable-egexo-grounding
RealSenseCamera 支持 depth
RobotController 支持 get_ee_pose()
debug 图保存
```

验收：

```bash
python scripts/inference.py \
  --model outputs/smolvla_realman/checkpoints/last/pretrained_model \
  --task "pick up the cube" \
  --freq 15 \
  --state-gripper-placeholder \
  --enable-egexo-grounding \
  --debug-egexo
```

此时可以仍然用普通 SmolVLA，只验证实时 grounding 计算不拖垮频率、不崩溃。

---

## Milestone 4：EgExoSmolVLA 最小训练版

完成：

```text
egexo_smolvla policy 注册
configuration_egexo_smolvla.py
modeling_egexo_smolvla.py
configs/egexo_smolvla_realman.yaml
```

第一版模型只要求：

```text
能读取 cam_high / cam_wrist / ego_roi / phase
能输出 7D action
能保存 checkpoint
能被 inference.py 加载
```

验收：

```bash
bash scripts/train.sh egexo_smolvla
```

和：

```bash
python scripts/evaluate_replay.py \
  --model outputs/egexo_smolvla_realman/checkpoints/last/pretrained_model \
  --dataset data/pick_cube_egexo_15fps
```

---

## Milestone 5：phase-aware action

完成：

```text
phase_gate
phase loss
phase embedding 或 dual action head
```

优先级：

```text
先做 phase embedding
再做 dual action head
```

不要一开始就强上双 action head，容易改崩 SmolVLA。

---

## Milestone 6：完整消融

完成：

```text
run_egexo_ablation.sh
所有 ablation 开关
日志记录
debug 可视化
```

至少输出：

```text
success_rate
near_contact_success_rate
final_alignment_error
grounding_valid_rate
roi_out_of_view_rate
inference_latency
failure_type
```

---

# 16. 风险控制

重点注意：

```text
1. 标定误差：projected ROI 只能作为 soft prior，不要硬裁剪唯一输入。
2. D455 depth 噪声：第一版只用 ROI median depth + 3D centroid。
3. DS87 视野窄：grounding invalid 时必须 fallback 到原始 wrist image。
4. 7D 接口不能破坏：qpos/action 永远保持 7D。
5. 旧 baseline 不能破坏：ACT、SmolVLA 普通训练和普通 inference 必须仍可用。
6. 不要把所有逻辑塞进 inference.py：几何和 target locator 必须放到 src/realman_vla/ 下。
```

---

# 17. 最低交付标准

最终请交付：

```text
configs/egexo_smolvla_realman.yaml
configs/calibration_realman.yaml
configs/egexo_runtime.yaml

src/realman_vla/geometry/crossview.py
src/realman_vla/geometry/calibration.py
src/realman_vla/vision/target_locator.py
src/realman_vla/policies/egexo_smolvla/...

scripts/check_egexo_hdf5.py
scripts/debug_project_grounding.py
scripts/run_egexo_ablation.sh
scripts/evaluate_replay.py

修改后的:
scripts/collect_data.py
scripts/collect_data_uarm.py
scripts/convert_to_lerobot.py
scripts/inference.py
scripts/train.sh
scripts/view_hdf5.py
```

最低验收：

```text
1. 旧 ACT / SmolVLA 数据转换、训练、推理仍然能跑
2. 新采集 HDF5 包含 RGB-D、ee_pose、calibration metadata
3. 新转换数据集包含 grounding.ego_roi、grounding.valid、phase
4. debug_project_grounding.py 能画出 exo ROI 和 ego projected ROI
5. inference.py 能实时计算 grounding 并保存 debug 图
6. egexo_smolvla 能训练并输出 7D action
7. egexo_smolvla checkpoint 能被 inference.py 加载
8. 能跑 naive dual-view concat vs egexo_smolvla 的核心对照
```

先完成数据和几何闭环，再做模型。不要反过来。






















<!-- ============================================================ -->
已按 collect_data_uarm.py 主线把 Ego-Exo 的第一阶段接进去了，重点是“不破坏旧接口、先把数据链路跑通”。

这次改动的核心有三块。第一，公共采集基础设施在 scripts/collect_data.py 里升级了：D435Camera 现在支持 enable_depth=True 的 RGB-D 采集，新增了 get_depth_frame() / get_intrinsics()，同时补了标定配置读取和 extract_ee_pose_from_realman_state()。第二，scripts/collect_data_uarm.py 和 scripts/collect_data.py 现在会附加保存 observations/depth/cam_high、observations/ee_pose、observations/grounding/* 占位字段，以及 metadata/cameras/cam_high/*、metadata/cameras/cam_wrist/* 标定元数据，旧的 qpos/action/cam_high/cam_wrist/timestamps 都保留不变。第三，scripts/convert_to_lerobot.py 新增了 --enable-egexo，会额外生成 observation.ee_pose、observation.grounding.ego_roi、observation.grounding.valid、observation.target_3d_base、observation.phase；同时我补了 scripts/check_egexo_hdf5.py、scripts/debug_project_grounding.py 和基础几何模块 src/realman_vla/geometry/crossview.py、src/realman_vla/geometry/calibration.py、src/realman_vla/vision/target_locator.py。

标定配置也已经落到 configs/calibration_realman.yaml、configs/egexo_runtime.yaml、configs/egexo_smolvla_realman.yaml。我把你给的 camera_intrinsics.yaml 和 T_base_camera.yaml/.txt 用到了 cam_high 上；cam_wrist 目前仍是占位内参 0 和 T_ee_cam = I，这是当前最大的已知假设，后面一旦你给我腕部相机内参和手眼外参，我可以把 grounding 精度那一段再收紧。

我已经用 python3 -m py_compile 过了这些新脚本和模块，语法是通的；但当前这个 shell 环境里缺 h5py，所以我没法在这里把自检脚本实际跑一遍。你现在可以直接试这几条：

python3 scripts/collect_data_uarm.py \
  --arm-ip 192.168.1.18 \
  --save-dir data/raw_hdf5 \
  --task-name pick_cube \
  --fps 15 \
  --calibration-config configs/calibration_realman.yaml
python3 scripts/check_egexo_hdf5.py --input data/raw_hdf5/pick_cube_0.hdf5
python3 scripts/convert_to_lerobot.py \
  --input-dir data/raw_hdf5 \
  --output-dir data/pick_cube_egexo_15fps \
  --repo-id local/pick_cube_egexo_15fps \
  --fps 15 \
  --task "pick up the cube" \
  --enable-egexo \
  --calib configs/calibration_realman.yaml \
  --runtime-config configs/egexo_runtime.yaml
如果你愿意，我下一步就继续做 Milestone 2，把 debug_project_grounding.py 和 inference.py 的实时 grounding 链路接完整。





Milestone 2 我已经继续推进了，重点把“几何投影先跑通”这条链路补实了。

现在的改动主要在三处。src/realman_vla/vision/target_locator.py 已经按文档做成了 TargetLocator，支持 manual_roi、color_heuristic 和预留的 external_detector，同时保留了 locate_target_roi() 兼容入口。src/realman_vla/geometry/crossview.py 现在会返回更完整的 reason、roi_out_of_view 和 projected_uv，能区分 invalid_exo_roi、invalid_depth、invalid_ego_intrinsics、projection_behind_camera 这类失败原因，不再只是一个布尔值。scripts/debug_project_grounding.py 也增强了：会优先用 HDF5 里已有的 target_roi_exo，否则退回 runtime locator；如果 configs/calibration_realman.yaml 里的内参无效，会尝试从 HDF5 metadata 回退读取；输出的 summary.csv 现在包含 roi_source、locator_reason、projection_reason、median_depth_m、target_x/y/z、exo/ego ROI 坐标这些诊断字段。

另外我在 src/realman_vla/geometry/calibration.py 补了 intrinsics_from_array() 和 has_valid_intrinsics()，方便调试脚本从 HDF5 的标定元数据回退恢复内参。相关文件已经通过 python3 -m py_compile，TargetLocator 的导入和基本调用也测过是通的。

一个很重要的现实提醒：你目前给我的标定只覆盖了顶部 exo 相机。configs/calibration_realman.yaml 里的 cam_wrist 还是占位内参 0 和单位外参，所以 debug_project_grounding.py 现在大概率会在 projection_reason 里看到 invalid_ego_intrinsics，这不是代码坏了，而是 wrist 标定还没补齐。等你把腕部相机的内参和 T_ee_cam 给我，我就能把这个链路从“可诊断”推进到“可用投影”。

你现在可以直接跑：

python3 scripts/debug_project_grounding.py \
  --input data/raw_hdf5/pick_cube_0.hdf5 \
  --calib configs/calibration_realman.yaml \
  --runtime-config configs/egexo_runtime.yaml \
  --output debug/grounding








inference.py 已经接上实时 grounding，Milestone 3 的主链路现在具备了。

这次我在 scripts/inference.py 做了这些关键改动：加了 --egexo-runtime-config、--calib、--enable-egexo-grounding、--disable-egexo-grounding、--debug-egexo 参数；把 RealSenseCamera 升级成可选 RGB-D；给 RobotController 增加了 get_ee_pose()；在主循环里于 preprocessor 之前插入了在线 exo ROI 定位、depth 到 3D、投影到 ego ROI、phase 估计的逻辑；并加了 EgExoDebugger，会在 --debug-egexo 下写 debug/egexo/step_*.jpg 和 debug/egexo/phase_log.csv。普通策略默认不吃额外 observation 键，只有模型真的声明需要这些特征时才会塞进 observation，所以现有 ACT / SmolVLA 路径不会被硬改崩。

我也顺手在 policy loader 里加了 egexo_smolvla 分支，但目前只是为后续留入口；真正的 EgExoSmolVLAPolicy 还没实现，所以现在 Milestone 3 的推荐验证方式仍然是用普通 smolvla 加 --enable-egexo-grounding。静态语法检查已经过了，但这个 shell 环境里没有 torch，所以我没法在这里把脚本完整 import 并实跑一遍。

直接验证可以用这条：

python3 scripts/inference.py \
  --model outputs/smolvla_realman/checkpoints/last/pretrained_model \
  --task "pick up the cube" \
  --freq 15 \
  --enable-egexo-grounding \
  --debug-egexo \
  --calib configs/calibration_realman.yaml \
  --egexo-runtime-config configs/egexo_runtime.yaml
一个很关键的限制还在：你当前 configs/calibration_realman.yaml 里的 cam_wrist 还是占位内参和单位外参，所以实时 grounding 现在大概率会跑通流程，但 grounding.valid 常常会是 0，或者 reason 会落在 ego 投影无效上。这是标定数据还不完整，不是推理链路没接上。你把腕部相机内参和 T_ee_cam 给我后，我下一步就可以继续把 Milestone 4 的 egexo_smolvla 最小训练版接起来。











Milestone 4 的最小训练版已经接上了，走的是“先能训起来、再逐步增强”的路线。

现在新增了 src/realman_vla/policies/egexo_smolvla/configuration_egexo_smolvla.py 和 src/realman_vla/policies/egexo_smolvla/modeling_egexo_smolvla.py。这个 EgExoSmolVLAPolicy 目前是阶段 A 的最小包装：继续复用 SmolVLA 主干，只对 wrist 图像按 ego_roi 做 soft mask 预处理，并容忍 ee_pose / grounding / phase 这些附加键存在，不去重写大模型。训练入口也补了专门适配器 src/realman_vla/policies/egexo_smolvla/train.py，会把 configs/egexo_smolvla_realman.yaml 翻译成当前 LeRobot 更稳妥能吃的 SmolVLA 训练配置，同时把 Ego-Exo 专属字段单独保存成 egexo_metadata.yaml。scripts/train.sh 也改了：现在会先导出 PYTHONPATH=src，并在 egexo_smolvla 时自动走这个适配器，所以你可以直接跑 bash scripts/train.sh egexo_smolvla。

我已经做过的验证是：这些新模块都通过了 python3 -m py_compile，配置翻译也能正常工作，egexo_smolvla_realman.yaml 会被正确降成 smolvla 训练配置并保留 Ego-Exo 元数据。还没法在这个 shell 里做端到端训练启动，因为这里缺 lerobot 和 torch，但代码已经按“有完整 conda 环境时可落地”的方式接好了。

你现在可以在你的训练环境里直接试：

bash scripts/train.sh egexo_smolvla
如果要断点续训：

bash scripts/train.sh egexo_smolvla resume
一个重要说明是：这个 Milestone 4 还是“最小训练版”，本质上仍然沿用 SmolVLA 主干训练，只把 Ego-Exo 的 ROI 先接进了训练前向，并没有上 dual action head 或 phase-aware mixing。下一步如果你要，我可以继续把这个包装器从阶段 A 往阶段 B 推，把 phase 真正做成 embedding 接进动作专家。















阶段 B 已经推进好了：phase 现在不只是标签，而是真正做成了 embedding 注入到动作专家的输入通道里。

具体来说，src/realman_vla/policies/egexo_smolvla/modeling_egexo_smolvla.py 现在在 EgExoSmolVLAPolicy.__init__() 里会初始化一个小的 phase_embedding MLP，把 observation.phase 从 1D 映射到和 observation.state 同维的向量，然后在 _prepare_egexo_batch() 里加到状态向量上。也就是说，这一版 phase 是作为额外条件直接注入 state 通道进入 SmolVLA 主干，属于你文档里“先做 phase embedding，再做 dual head”的稳妥路线。为了避免无效 grounding 干扰，这个 phase embedding 还会受 observation.grounding.valid 门控，invalid 时自动压掉。

配置侧我也同步补了 src/realman_vla/policies/egexo_smolvla/configuration_egexo_smolvla.py 和 configs/egexo_smolvla_realman.yaml 里的新字段：
use_phase_embedding: true
phase_embedding_dim: 32
phase_embedding_scale: 0.25

另外我把训练闭环再补了一步：src/realman_vla/policies/egexo_smolvla/train.py 现在会在训练结束后自动 patch 各 checkpoint 的 config.json，把 type 改回 egexo_smolvla，并把 Ego-Exo 元信息写进去。这样后续用 scripts/inference.py 加载 checkpoint 时，就不会又被识别成普通 smolvla 了。

我已经验证过这些改动能通过 py_compile，配置翻译也能正确读出新的 phase embedding 参数。下一步如果你愿意，我建议直接进入你文档里的 Milestone 5 后半段，把 phase_gate / phase loss 也补上；如果想更稳一点，也可以先做一轮最小训练冒烟测试，确认 bash scripts/train.sh egexo_smolvla 在你的 conda 环境里能进 dataloader 和第一个 forward。















phase_gate 和 phase loss 已经补上了，而且还是用比较稳的外挂方式接进现有包装器，没有去硬拆 SmolVLA 主干。

现在 src/realman_vla/policies/egexo_smolvla/modeling_egexo_smolvla.py 里新增了一个小的 phase_gate 头：输入是当前 observation.state，再加上可选的 observation.ee_pose，输出 phase_logits 和 phase_probs。训练时如果 batch 里有 observation.phase，就会用 binary_cross_entropy_with_logits 算 phase_loss，再按 configs/egexo_smolvla_realman.yaml 里的 loss.phase_weight 叠加到主 loss 上。如果 observation.grounding.valid 存在，还会用它对 phase loss 做门控，避免 invalid grounding 帧把监督弄脏。

配置也同步补了两个阶段 B/C 过渡参数：
phase_gate_hidden_dim: 64
phase_gate_use_ee_pose: true

而且 phase embedding 那部分还保留着，所以现在这条 EgExoSmolVLA 路线已经同时具备：

wrist ROI soft mask
phase embedding 注入 state
phase gate 输出 phase_probs
phase loss 叠加训练





























双 action head 和 action = p_t * a_t + p_c * a_c 的 mixing 已经接上了。

现在 src/realman_vla/policies/egexo_smolvla/modeling_egexo_smolvla.py 里新增了两条外挂头：
transport_action_head
contact_action_head

它们基于当前条件特征分别预测 transport_action 和 contact_action，然后直接用 phase_gate 的输出做 soft mixing：

p_contact = phase_head_output["phase_probs"]
p_transport = 1.0 - p_contact
mixed_action = p_transport * transport_action + p_contact * contact_action
训练时，如果 batch 里有 action，现在会额外计算 dual_action_loss，并按配置里的 loss.dual_action_weight 叠加到总 loss 上；推理时，如果 select_action() 拿到的是 action tensor，就会直接返回 mixed_action，如果拿到的是 dict，也会把 action 相关字段替换成 mixed 版本。也就是说，这条路线现在已经具备：

phase_embedding
phase_gate
phase_loss
transport/contact 双 action head
p_t / p_c soft mixing
配置侧也同步补了：
configs/egexo_smolvla_realman.yaml

里面新增了：
use_dual_action_head: true
dual_action_head_hidden_dim: 128
dual_action_head_use_ee_pose: true
dual_action_weight: 0.1

这些代码已经通过了 py_compile，配置翻译层也确认能读到这些新参数。

下一步最值得做的是一次真实训练环境里的冒烟验证，确认：

bash scripts/train.sh egexo_smolvla
至少能进第一个 batch 和第一个 forward。再往后就可以进入你文档里的 Milestone 6，把 run_egexo_ablation.sh 和完整 ablation 跑法补齐。
































消融流程已经按正常可复用的方式补上了，而且不依赖机械臂。

现在新增了两个核心脚本：
scripts/run_egexo_ablation.sh
scripts/evaluate_replay.py

run_egexo_ablation.sh 会基于 configs/egexo_smolvla_realman.yaml 自动生成一组消融配置，并把输出目录规范成：
outputs/ablation/cam_high_only
outputs/ablation/cam_wrist_only
outputs/ablation/naive_concat
outputs/ablation/asymmetric
outputs/ablation/grounding
outputs/ablation/phase_head
outputs/ablation/full
outputs/ablation/full_no_aug

同时我把 ablation 段也正式加进了 configs/egexo_smolvla_realman.yaml，并让训练适配器 src/realman_vla/policies/egexo_smolvla/train.py 会把这些消融元信息一起保存到 egexo_metadata.yaml。scripts/train.sh 也补了 PYTHONPATH 打印，方便后续排查环境问题。

evaluate_replay.py 是离线 sanity check。它支持两种输入：

原始 HDF5 文件或目录
LeRobot 数据集根目录
对原始 HDF5，它会检查 7D qpos/action、grounding_valid_rate、phase_distribution 和 action shape 一致性。对 LeRobot 数据集，它会先检查 meta/info.json 里的 feature 定义；如果环境里有 pyarrow，还会进一步读 parquet 统计 grounding_valid_rate 和 phase_distribution。这样你现在没有机械臂，也能先验证数据和消融配置组织是通的。

我已经实际验证过这几件事：

scripts/run_egexo_ablation.sh 能成功生成 manifest 和所有变体配置。
scripts/evaluate_replay.py 和训练适配器都通过了 py_compile。
run_egexo_ablation.sh 已经设成可执行。
你现在可以先这样用：

bash scripts/run_egexo_ablation.sh configs/egexo_smolvla_realman.yaml print
看生成哪些变体。

跑全部消融时：

bash scripts/run_egexo_ablation.sh configs/egexo_smolvla_realman.yaml run
离线检查数据时：

python3 scripts/evaluate_replay.py --input data/pick_cube_egexo_15fps
或者检查原始 HDF5：

python3 scripts/evaluate_replay.py --input data/raw_hdf5
需要说明的一点是：我现在补的是“实验组织和离线检查流程”，不是实机成功率评测；像 success_rate / final_alignment_error / failure_type / inference_latency 这种最终指标，还是要等你后面接上机械臂或至少有可复现的 replay/eval 环境后再补真实统计。现在这套脚本已经把前面的准备工作都铺好了。