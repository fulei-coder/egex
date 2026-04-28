#!/usr/bin/env python3
"""
通用多策略推理脚本 — RealMan RM65

自动检测策略类型 (ACT / Diffusion / VQ-BeT / SmolVLA / Pi0 / Pi0.5)
加载预处理/后处理管线，实时控制机械臂。

特性:
  - 自动从 config.json 检测策略类型
  - EMA 动作平滑 + 死区过滤（减少抖动）
  - 异步夹爪控制（Modbus 写入不阻塞主循环）
  - ACT: Temporal Ensemble 支持
  - VLA: Language instruction 支持

用法:
  # ACT
  python scripts/inference.py \\
      --model outputs/act_realman/checkpoints/100000/pretrained_model \\
      --arm-ip 192.168.1.18 --freq 15

  # SmolVLA (需要 task 描述)
  python scripts/inference.py \\
      --model outputs/smolvla_realman/checkpoints/50000/pretrained_model \\
      --task "pick up the cube" --freq 15

  # Pi0 (离线模式，不下载HuggingFace)
  python scripts/inference.py \\
      --model outputs/pi0_realman/checkpoints/30000/pretrained_model \\
      --task "pick up the cube" --offline
"""

import torch
import numpy as np
import time
import sys
import threading
import cv2
import json
import argparse
import csv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml
from realman_vla.geometry.calibration import (
    ee_pose_to_T_base_ee,
    intrinsics_from_config,
    load_calibration_config,
    transform_from_config,
)
from realman_vla.geometry.crossview import project_exo_roi_to_ego
from realman_vla.vision.target_locator import TargetLocator

# ============ 默认硬件配置（根据你的硬件修改） ============
DEFAULT_ARM_IP = "192.168.1.18"           # 出厂默认 192.168.2.18
DEFAULT_ARM_PORT = 8080
DEFAULT_CAM_TOP_SERIAL = "346122070612"  #435
# DEFAULT_CAM_TOP_SERIAL = "108222250854"     #455  # rs-enumerate-devices | grep Serial

DEFAULT_DS87_RGB_TOPIC = "/Scepter/color/image_raw"

RM_SDK_PATH = "/home/tony/RM_API2/Python"
if RM_SDK_PATH not in sys.path:
    sys.path.append(RM_SDK_PATH)

# Modbus 夹爪参数
GRIPPER_MODBUS_PORT = 1
GRIPPER_MODBUS_ADDR = 43
GRIPPER_MODBUS_DEVICE = 1
GRIPPER_MODBUS_NUM = 2

# 训练数据典型起始位姿（根据你的数据修改）
INIT_POSE = np.array([-15.0, 3.0, 89.0, 0.5, 86.0, -15.0, 1.0], dtype=np.float32)


# ============ 版本兼容性补丁 ============
def _patch_config_compat(config_path):
    """清理 config.json 中当前版本 LeRobot 不支持的字段"""
    import dataclasses
    with open(config_path) as f:
        cfg = json.load(f)

    # 尝试获取对应策略的有效字段
    policy_type = cfg.get("type", "")
    valid_fields = set()

    try:
        if policy_type == "smolvla":
            from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
            valid_fields = {f.name for f in dataclasses.fields(SmolVLAConfig)}
        from lerobot.configs.policies import PreTrainedConfig
        valid_fields |= {f.name for f in dataclasses.fields(PreTrainedConfig)}
    except ImportError:
        return

    removed = []
    for key in list(cfg.keys()):
        if key not in valid_fields and key != "type":
            removed.append(key)
            del cfg[key]

    if removed:
        print(f"  ⚠️  过滤不兼容字段: {removed}")
        with open(config_path, 'w') as f:
            json.dump(cfg, f, indent=4)


# ============ 策略加载器 ============
def load_policy(model_path: str, device: torch.device):
    """自动检测策略类型并加载，返回 (policy, policy_type)"""
    model_path = Path(model_path)
    config_path = model_path / "config.json"

    if not config_path.exists():
        raise FileNotFoundError(f"找不到配置文件: {config_path}")

    with open(config_path) as f:
        config = json.load(f)

    policy_type = config.get("type", "unknown")
    print(f"检测到策略类型: {policy_type}")

    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(str(model_path))
        # ACT 特有: Temporal Ensemble 平滑
        if hasattr(policy.config, 'temporal_ensemble_coeff'):
            policy.config.temporal_ensemble_coeff = 0.01
            from lerobot.policies.act.modeling_act import ACTTemporalEnsembler
            policy.temporal_ensembler = ACTTemporalEnsembler(
                temporal_ensemble_coeff=0.01,
                chunk_size=policy.config.chunk_size
            )
            print(f"  Temporal Ensemble: Enabled (coeff=0.01)")

    elif policy_type == "vqbet":
        from lerobot.policies.vqbet.modeling_vqbet import VQBeTPolicy
        policy = VQBeTPolicy.from_pretrained(str(model_path))

    elif policy_type == "diffusion":
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        policy = DiffusionPolicy.from_pretrained(str(model_path))

    elif policy_type == "smolvla":
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        _patch_config_compat(config_path)
        policy = SmolVLAPolicy.from_pretrained(str(model_path))

    elif policy_type == "egexo_smolvla":
        try:
            from realman_vla.policies.egexo_smolvla.modeling_egexo_smolvla import EgExoSmolVLAPolicy
        except ImportError as exc:
            raise ImportError(
                "检测到 egexo_smolvla，但本地策略实现尚未安装完成。"
                "Milestone 3 可以先用普通 smolvla + --enable-egexo-grounding 验证实时 grounding。"
            ) from exc
        policy = EgExoSmolVLAPolicy.from_pretrained(str(model_path))

    elif policy_type == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(str(model_path))

    elif policy_type == "pi05":
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy
        policy = PI05Policy.from_pretrained(str(model_path))

    else:
        raise ValueError(f"不支持的策略类型: {policy_type}")

    policy.to(device)
    policy.eval()
    return policy, policy_type


# ============ 辅助函数 ============
def dec_to_register(dec):
    """小数(0~1) → Modbus 寄存器值"""
    value = dec * 256000
    R0 = int(value // (256 ** 3))
    remainder = value % (256 ** 3)
    R1 = int(remainder // (256 ** 2))
    remainder = remainder % (256 ** 2)
    R2 = int(remainder // 256)
    R3 = int(remainder % 256)
    return [R0, R1, R2, R3]


def register_to_dec(register_value):
    """Modbus 寄存器值 → 小数(0~1)"""
    return (register_value[0] * 256**3 + register_value[1] * 256**2 +
            register_value[2] * 256 + register_value[3]) / 256000


def _get_feature_dim(feature):
    shape = getattr(feature, 'shape', None)
    if shape is None:
        return None
    try:
        if len(shape) == 0:
            return None
        return int(shape[-1])
    except Exception:
        return None


def extract_ee_pose_from_realman_state(state):
    if not isinstance(state, dict):
        return np.zeros(6, dtype=np.float32)

    for key in ("pose", "tool_pose", "tcp_pose", "end_pose"):
        value = state.get(key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.size >= 6:
            return arr[:6].copy()

    for container_key in ("arm_state", "current_state", "state"):
        nested = state.get(container_key)
        if not isinstance(nested, dict):
            continue
        for key in ("pose", "tool_pose", "tcp_pose"):
            value = nested.get(key)
            if value is None:
                continue
            arr = np.asarray(value, dtype=np.float32).reshape(-1)
            if arr.size >= 6:
                return arr[:6].copy()

    return np.zeros(6, dtype=np.float32)


def estimate_phase(ee_pose, target_3d_base, valid, runtime_cfg):
    if not valid:
        return 0.0
    threshold = float((runtime_cfg.get("phase") or {}).get("distance_threshold_m", 0.08))
    ee_pose = np.asarray(ee_pose, dtype=np.float32).reshape(-1)
    target_3d_base = np.asarray(target_3d_base, dtype=np.float32).reshape(-1)
    if ee_pose.size < 3 or target_3d_base.size < 3:
        return 0.0
    distance = float(np.linalg.norm(ee_pose[:3] - target_3d_base[:3]))
    return 0.0 if distance > threshold else 1.0


class EgExoDebugger:
    def __init__(self, cfg, enabled=False):
        debug_cfg = (cfg or {}).get("debug") or {}
        self.enabled = bool(enabled)
        self.save_every_n_steps = int(debug_cfg.get("save_every_n_steps", 10))
        self.draw_exo_roi = bool(debug_cfg.get("draw_exo_roi", True))
        self.draw_ego_projected_roi = bool(debug_cfg.get("draw_ego_projected_roi", True))
        self.save_dir = Path(debug_cfg.get("save_dir", "debug/egexo"))
        self.rows = []
        self.csv_path = self.save_dir / "phase_log.csv"
        if self.enabled:
            self.save_dir.mkdir(parents=True, exist_ok=True)

    def _draw_roi(self, image_bgr, roi_xyxy, label):
        canvas = image_bgr.copy()
        roi = np.asarray(roi_xyxy, dtype=np.int32).reshape(4)
        if roi[2] > roi[0] and roi[3] > roi[1]:
            cv2.rectangle(canvas, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 0), 2)
        cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return canvas

    def log(self, step, grounding, phase, exo_roi, exo_image_bgr, ego_image_bgr):
        if not self.enabled:
            return

        row = {
            "step": int(step),
            "time": time.time(),
            "grounding_valid": float(grounding.get("valid", False)),
            "phase_transport": 1.0 - float(phase),
            "phase_contact": float(phase),
            "target_x": float(np.asarray(grounding.get("target_3d_base", np.zeros(3)))[0]),
            "target_y": float(np.asarray(grounding.get("target_3d_base", np.zeros(3)))[1]),
            "target_z": float(np.asarray(grounding.get("target_3d_base", np.zeros(3)))[2]),
            "ego_roi_x1": float(np.asarray(grounding.get("ego_roi_xyxy", np.zeros(4)))[0]),
            "ego_roi_y1": float(np.asarray(grounding.get("ego_roi_xyxy", np.zeros(4)))[1]),
            "ego_roi_x2": float(np.asarray(grounding.get("ego_roi_xyxy", np.zeros(4)))[2]),
            "ego_roi_y2": float(np.asarray(grounding.get("ego_roi_xyxy", np.zeros(4)))[3]),
            "reason": str(grounding.get("reason", "unknown")),
        }
        self.rows.append(row)

        if self.rows:
            with self.csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(self.rows[0].keys()))
                writer.writeheader()
                writer.writerows(self.rows)

        if self.save_every_n_steps <= 0 or step % self.save_every_n_steps != 0:
            return

        if self.draw_exo_roi:
            exo_canvas = self._draw_roi(
                exo_image_bgr,
                exo_roi,
                f"exo valid={int(bool(grounding.get('valid', False)))} {grounding.get('reason', 'unknown')}",
            )
            cv2.imwrite(str(self.save_dir / f"step_{step:06d}_exo.jpg"), exo_canvas)

        if self.draw_ego_projected_roi:
            ego_canvas = self._draw_roi(
                ego_image_bgr,
                grounding.get("ego_roi_xyxy", np.zeros(4, dtype=np.float32)),
                f"ego phase={phase:.1f}",
            )
            cv2.imwrite(str(self.save_dir / f"step_{step:06d}_ego.jpg"), ego_canvas)


# ============ RealSense 相机 ============
import pyrealsense2 as rs


class RealSenseCamera:
    """RealSense 相机异步采集"""

    def __init__(self, serial_number, width=640, height=480, fps=30, enable_depth=False):
        self.serial_number = str(serial_number)
        self.width, self.height = width, height
        self.enable_depth = bool(enable_depth)
        self.latest_color = np.zeros((height, width, 3), dtype=np.uint8)
        self.latest_depth = np.zeros((height, width), dtype=np.uint16)
        self.lock = threading.Lock()
        self.stopped = False
        self.intrinsics = None
        self.depth_scale = 0.0
        self.align = None

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        if self.enable_depth:
            self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            self.align = rs.align(rs.stream.color)

        try:
            profile = self.pipeline.start(self.config)
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, 150)

            try:
                color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
                color_intr = color_profile.get_intrinsics()
                self.intrinsics = {
                    "fx": float(color_intr.fx),
                    "fy": float(color_intr.fy),
                    "cx": float(color_intr.ppx),
                    "cy": float(color_intr.ppy),
                    "width": int(color_intr.width),
                    "height": int(color_intr.height),
                    "depth_scale": 0.0,
                }
                if self.enable_depth:
                    depth_sensor = profile.get_device().first_depth_sensor()
                    self.depth_scale = float(depth_sensor.get_depth_scale())
                    self.intrinsics["depth_scale"] = self.depth_scale
            except Exception as exc:
                print(f"[WARN] RealSense intrinsics unavailable: {exc}")

            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            mode = "RGB-D" if self.enable_depth else "RGB"
            print(f"[✓] Camera {self.serial_number} OK ({mode})")
        except Exception as e:
            print(f"[✗] Camera {self.serial_number}: {e}")
            raise

    def _update_loop(self):
        while not self.stopped:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                if self.enable_depth and self.align is not None:
                    frames = self.align.process(frames)
                color_frame = frames.get_color_frame()
                depth_frame = frames.get_depth_frame() if self.enable_depth else None
                if color_frame:
                    frame_data = np.asanyarray(color_frame.get_data())
                    with self.lock:
                        self.latest_color = frame_data.copy()
                        if depth_frame:
                            depth_data = np.asanyarray(depth_frame.get_data())
                            if depth_data.shape == self.latest_depth.shape:
                                self.latest_depth = depth_data.copy()
            except Exception:
                pass

    def get_frame(self):
        with self.lock:
            return self.latest_color.copy()

    def get_depth_frame(self):
        with self.lock:
            return self.latest_depth.copy()

    def get_intrinsics(self):
        with self.lock:
            intrinsics = dict(self.intrinsics or {})
            if not intrinsics:
                intrinsics = {
                    "fx": 0.0,
                    "fy": 0.0,
                    "cx": 0.0,
                    "cy": 0.0,
                    "width": self.width,
                    "height": self.height,
                    "depth_scale": self.depth_scale,
                }
            return intrinsics

    def close(self):
        self.stopped = True
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ============ ROS 2 初始化 ============
_ros_init_lock = threading.Lock()
_ros_inited = False
_ros_node = None

def _ros_spin_loop(node):
    import rclpy
    try:
        rclpy.spin(node)
    except Exception as e:
        print(f"[ROS2] Spin exited: {e}")

def ensure_ros_node(node_name="lerobot_ds87_rgb_collector"):
    global _ros_inited, _ros_node
    with _ros_init_lock:
        if _ros_inited:
            return

        try:
            import yaml  # noqa: F401
        except Exception as e:
            raise RuntimeError(
                "ROS Python 环境缺少 yaml 模块。请先执行: python -m pip install PyYAML"
            )

        import rclpy
        if not rclpy.ok():
            rclpy.init(args=None)
        
        # ROS 2 需要显式的 Node 对象
        _ros_node = rclpy.create_node(node_name)
        
        # 启动后台处理回调的 spin 线程
        spin_thread = threading.Thread(target=_ros_spin_loop, args=(_ros_node,), daemon=True)
        spin_thread.start()
        
        _ros_inited = True


# ============ DS87 ROS RGB 相机模块============
class DS87RosCamera:

    def __init__(self, topic=DEFAULT_DS87_RGB_TOPIC, width=640, height=480):
        self.topic = topic
        self.width = width
        self.height = height
        self.target_shape = (height, width, 3)

        self.latest_color = np.zeros(self.target_shape, dtype=np.uint8)
        self.lock = threading.Lock()
        self.is_active = False
        self.frame_count = 0
        self.last_frame_time = 0.0
        self.last_warn_time = 0.0
        self.sub = None

        try:
            ensure_ros_node()

            import rclpy
            from sensor_msgs.msg import Image
            from rclpy.qos import QoSProfile, QoSHistoryPolicy, QoSReliabilityPolicy
            
            global _ros_node

            qos_profile = QoSProfile(
                history=QoSHistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=QoSReliabilityPolicy.BEST_EFFORT
            )

            self.sub = _ros_node.create_subscription(
                Image,
                self.topic,
                self._callback,
                qos_profile
            )
            self.is_active = True
            print(f"[DS87-ROS2] subscribed topic: {self.topic}")

            time.sleep(2.0)
            if self.frame_count == 0:
                print(
                    f"[!] DS87-ROS2 subscribed but no image received from topic: {self.topic}\n"
                    f"    请检查: ros2 topic hz {self.topic}"
                )

        except Exception as e:
            print(f"[!] DS87-ROS2 init failed: {e}")
            print("[!] 请确认:")
            print("    1) 已 source ROS 2 工作空间")
            print("    2) 已启动相机节点")
            print(f"    3) 话题存在: {self.topic}")
            self.is_active = False

    def _safe_warn(self, msg, interval_sec=2.0):
        now = time.time()
        if now - self.last_warn_time >= interval_sec:
            print(msg)
            self.last_warn_time = now

    def _decode_ros_image(self, msg):

        h = int(msg.height)
        w = int(msg.width)
        enc = (msg.encoding or "").lower()
        step = int(msg.step)
        data = np.frombuffer(msg.data, dtype=np.uint8)

        if h <= 0 or w <= 0:
            return None

        # bgr8
        if enc == "bgr8":
            expected = h * step
            if data.size < expected:
                return None
            img = data[:expected].reshape((h, step))
            img = img[:, :w * 3].reshape((h, w, 3))
            return img.copy()

        # rgb8
        if enc == "rgb8":
            expected = h * step
            if data.size < expected:
                return None
            img = data[:expected].reshape((h, step))
            img = img[:, :w * 3].reshape((h, w, 3))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img.copy()

        # bgra8
        if enc == "bgra8":
            expected = h * step
            if data.size < expected:
                return None
            img = data[:expected].reshape((h, step))
            img = img[:, :w * 4].reshape((h, w, 4))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img.copy()

        # rgba8
        if enc == "rgba8":
            expected = h * step
            if data.size < expected:
                return None
            img = data[:expected].reshape((h, step))
            img = img[:, :w * 4].reshape((h, w, 4))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            return img.copy()

        # mono8
        if enc == "mono8":
            expected = h * step
            if data.size < expected:
                return None
            img = data[:expected].reshape((h, step))
            img = img[:, :w].reshape((h, w))
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            return img.copy()

        # 未知编码：尝试按 3 通道裸解析
        expected = h * w * 3
        if data.size >= expected:
            try:
                img = data[:expected].reshape((h, w, 3))
                return img.copy()
            except Exception:
                pass

        return None

    def _callback(self, msg):
        try:
            img = self._decode_ros_image(msg)
            if img is None:
                self._safe_warn(
                    f"[DS87-ROS2] unsupported/invalid image: encoding={msg.encoding}, "
                    f"size=({msg.height},{msg.width}), step={msg.step}",
                    interval_sec=2.0
                )
                return

            if img.shape[:2] != (self.height, self.width):
                img = cv2.resize(img, (self.width, self.height))

            if img.dtype != np.uint8:
                img = np.clip(img, 0, 255).astype(np.uint8)

            with self.lock:
                self.latest_color = img.copy()
                self.frame_count += 1
                self.last_frame_time = time.time()

            if self.frame_count == 1:
                try:
                    cv2.imwrite("/tmp/ds87_ros_first_frame.jpg", img)
                    print(
                        f"[DS87-ROS2] first RGB frame saved: "
                        f"/tmp/ds87_ros_first_frame.jpg, "
                        f"encoding={msg.encoding}, shape={img.shape}, mean={img.mean():.2f}"
                    )
                except Exception as e:
                    print(f"[DS87-ROS2] save first frame failed: {e}")

        except Exception as e:
            self._safe_warn(f"[DS87-ROS2] callback error: {e}", interval_sec=1.0)

    def get_frame(self):
        with self.lock:
            frame = self.latest_color.copy()

        if float(frame.mean()) < 3.0 and self.is_active:
            self._safe_warn(
                f"[WARN] DS87-ROS2 frame looks dark, frame_count={self.frame_count}, topic={self.topic}",
                interval_sec=2.0
            )
        return frame

    def get_status(self):
        age = time.time() - self.last_frame_time if self.last_frame_time > 0 else None
        return {
            "is_active": self.is_active,
            "frame_count": self.frame_count,
            "last_frame_age": age,
            "topic": self.topic,
        }

    def close(self):
        try:
            if self.sub is not None:
                global _ros_node
                _ros_node.destroy_subscription(self.sub)
                self.sub = None
        except Exception:
            pass


# ============ 机械臂控制器 ============
class RobotController:
    """机械臂控制（支持 EMA 平滑 + 死区过滤 + 异步夹爪）

    关键设计:
      - EMA (Exponential Moving Average): 平滑关节角指令，减少抖动
      - 死区过滤: 变化小于阈值时不发送，减少通信压力
      - 异步夹爪: Modbus 写入在独立线程，不阻塞主控制循环
    """

    def __init__(self, arm, gripper_params):
        self.arm = arm
        self.gripper_params = gripper_params
        self.lock = threading.Lock()

        self._last_gripper_cmd = None
        self._cached_gripper_pos = 0.5
        self._smoothed_action = None
        self._last_joint_cmd = None
        # self.ema_alpha = 1       # EMA 系数 (0.3=平滑, 0.7=响应快)
        # self.joint_deadzone = 10.0  # 死区阈值(度)
        self.ema_alpha = 0.3       # EMA 系数 (0.3=平滑, 0.7=响应快)
        self.joint_deadzone = 0.5  # 死区阈值(度)

    def get_qpos(self, include_gripper=False):
        """获取当前状态。
        - include_gripper=False: 返回 6 维关节角
        - include_gripper=True : 返回 7 维 [6关节 + 夹爪]
        """
        with self.lock:
            joint_state = self.arm.rm_get_current_arm_state()
            joint_angles = joint_state[1]['joint'][:6]

            if not include_gripper:
                return np.array(joint_angles, dtype=np.float32)

            gripper_pos = self._cached_gripper_pos
            try:
                ret, gripper_reg = self.arm.rm_read_multiple_holding_registers(self.gripper_params)
                if ret == 0 and gripper_reg:
                    gripper_pos = 1 - register_to_dec(gripper_reg)
                    self._cached_gripper_pos = gripper_pos
            except Exception:
                pass

            return np.array(joint_angles + [gripper_pos], dtype=np.float32)

    def get_ee_pose(self):
        with self.lock:
            code, state = self.arm.rm_get_current_arm_state()
        if code != 0:
            return np.zeros(6, dtype=np.float32)
        return extract_ee_pose_from_realman_state(state)

    def set_qpos(self, qpos, use_smoothing=True, execute_gripper=False):
        """设置目标关节角；可选执行第7维夹爪控制。"""
        with self.lock:
            joint_target = qpos[:6].copy()

            # EMA 平滑
            if use_smoothing:
                if self._smoothed_action is None:
                    self._smoothed_action = joint_target.copy()
                else:
                    self._smoothed_action = (
                        self.ema_alpha * joint_target +
                        (1 - self.ema_alpha) * self._smoothed_action
                    )
                joint_target = self._smoothed_action.copy()

            # 死区过滤
            should_send = True
            if self._last_joint_cmd is not None:
                max_delta = np.max(np.abs(joint_target - self._last_joint_cmd))
                if max_delta < self.joint_deadzone:
                    should_send = False

            if should_send:   #speed
                self.arm.rm_movej(joint_target.tolist(), 5, 0, 0, 0)
                self._last_joint_cmd = joint_target.copy()

            # 只有显式启用时才执行第7维夹爪控制
            if execute_gripper and len(qpos) > 6:
                gripper_binary = 1 if qpos[6] > 0.5 else 0
                if self._last_gripper_cmd != gripper_binary:
                    self._last_gripper_cmd = gripper_binary
                    threading.Thread(
                        target=self._try_write_gripper,
                        args=(gripper_binary,),
                        daemon=True
                    ).start()

    def _try_write_gripper(self, gripper_binary):
        """异步夹爪写入（失败静默）"""
        try:
            gripper_target = 1.0 - float(gripper_binary)
            gripper_reg = dec_to_register(gripper_target)
            write_params = type(self.gripper_params)(
                port=self.gripper_params.port,
                address=self.gripper_params.address,
                device=self.gripper_params.device,
                num=self.gripper_params.num
            )
            self.arm.rm_write_registers(write_params, gripper_reg)
        except Exception:
            pass

    def move_to_init(self, init_pose):
        """移动到初始位姿（阻塞）"""
        print(f"移动到初始位姿...")
        self.arm.rm_movej(init_pose[:6].tolist(), 80, 0, 0, 1)

        gripper_reg = dec_to_register(1.0 - init_pose[6])
        write_params = type(self.gripper_params)(
            port=self.gripper_params.port,
            address=self.gripper_params.address,
            device=self.gripper_params.device,
            num=self.gripper_params.num
        )
        self.arm.rm_write_registers(write_params, gripper_reg)
        time.sleep(1.0)

    def stop(self):
        try:
            self.arm.rm_set_arm_stop()
        except Exception:
            pass

    def close(self):
        self.arm.rm_delete_robot_arm()


# ============ 主推理循环 ============
def main():
    parser = argparse.ArgumentParser(description='RealMan 通用策略推理')
    parser.add_argument('--model', type=str, required=True,
                        help='模型路径 (如 outputs/act/checkpoints/100000/pretrained_model)')
    parser.add_argument('--arm-ip', type=str, default=DEFAULT_ARM_IP, help='机械臂IP')
    parser.add_argument('--arm-port', type=int, default=DEFAULT_ARM_PORT, help='机械臂端口')
    parser.add_argument('--cam-top', type=str, default=DEFAULT_CAM_TOP_SERIAL, help='顶部相机序列号')
    parser.add_argument('--cam-wrist-topic', type=str, default=DEFAULT_DS87_RGB_TOPIC, help='腕部相机ROS2话题')
    parser.add_argument('--freq', type=float, default=15.0,
                        help='控制频率(Hz)，应与训练数据fps一致')
    parser.add_argument('--task', type=str, default='pick up the cube',
                        help='VLA 任务描述 (Pi0/SmolVLA 需要)')
    parser.add_argument('--state-with-gripper', action='store_true',
                        help='观测状态包含夹爪维度（7维）')
    parser.add_argument('--state-gripper-placeholder', action='store_true',
                        help='观测状态使用 7 维夹爪占位值，不读取 Modbus 夹爪状态')
    parser.add_argument('--gripper-placeholder-value', type=float, default=0.0,
                        help='观测状态第7维的夹爪占位值')
    parser.add_argument('--execute-gripper', action='store_true',
                        help='执行 action 第7维夹爪控制（默认关闭）')
    parser.add_argument('--egexo-runtime-config', type=str, default='configs/egexo_runtime.yaml',
                        help='Ego-Exo runtime 配置文件')
    parser.add_argument('--calib', type=str, default='configs/calibration_realman.yaml',
                        help='相机标定配置文件')
    parser.add_argument('--enable-egexo-grounding', action='store_true',
                        help='对普通策略也启用实时 Ego-Exo grounding 与 debug')
    parser.add_argument('--disable-egexo-grounding', action='store_true',
                        help='显式关闭实时 Ego-Exo grounding')
    parser.add_argument('--debug-egexo', action='store_true',
                        help='保存 Ego-Exo debug 图和 phase_log.csv')
    parser.add_argument('--headless', action='store_true', help='无GUI模式')
    parser.add_argument('--offline', action='store_true', help='离线模式 (不从HuggingFace下载)')
    args = parser.parse_args()

    if args.offline:
        import os
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'

    # 导入机械臂SDK
    from Robotic_Arm.rm_robot_interface import (
        RoboticArm, rm_thread_mode_e, rm_peripheral_read_write_params_t
    )
    from lerobot.processor.pipeline import DataProcessorPipeline

    print("=" * 50)
    print("  RealMan RM65 通用策略推理")
    print("=" * 50)

    # 1. 加载模型
    print("\n[1/5] Loading policy...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy, policy_type = load_policy(args.model, device)
    print(f"      Device: {device}")

    runtime_cfg = load_calibration_config(args.egexo_runtime_config)
    calib_cfg = load_calibration_config(args.calib)
    enable_egexo = (
        policy_type == "egexo_smolvla"
        or args.enable_egexo_grounding
    ) and not args.disable_egexo_grounding

    preprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_preprocessor.json')
    postprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_postprocessor.json')

    # 检测输入特征
    input_features = policy.config.input_features
    use_cam_high = 'observation.images.cam_high' in input_features
    use_cam_wrist = 'observation.images.cam_wrist' in input_features
    needs_ee_pose = 'observation.ee_pose' in input_features
    needs_grounding_roi = 'observation.grounding.ego_roi' in input_features
    needs_grounding_valid = 'observation.grounding.valid' in input_features
    needs_target_3d = 'observation.target_3d_base' in input_features
    needs_phase = 'observation.phase' in input_features
    attach_egexo_features = (
        policy_type == "egexo_smolvla"
        or needs_ee_pose
        or needs_grounding_roi
        or needs_grounding_valid
        or needs_target_3d
        or needs_phase
    )
    print(f"      Input features: {list(input_features.keys())}")
    print(f"      Ego-Exo grounding: {'ON' if enable_egexo else 'OFF'}")

    state_feature = input_features.get('observation.state')
    state_dim = _get_feature_dim(state_feature)
    if state_dim is None:
        raise RuntimeError("无法读取模型 observation.state 的维度，请检查模型配置")
    if state_dim not in (6, 7):
        raise RuntimeError(
            f"当前推理脚本仅支持 6D/7D observation.state，模型为 {state_dim}D。"
            "6D 对应 collect_data.py，7D 对应 collect_data_uarm.py。"
        )

    output_features = getattr(policy.config, 'output_features', {})
    action_feature = output_features.get('action') if output_features is not None else None
    action_dim = _get_feature_dim(action_feature)
    if action_dim is not None and action_dim not in (6, 7):
        raise RuntimeError(
            f"当前推理脚本仅支持 6D/7D action，模型为 {action_dim}D。"
            "6D 对应 collect_data.py，7D 对应 collect_data_uarm.py。"
        )

    if args.state_gripper_placeholder:
        state_mode = 'placeholder'
    elif args.state_with_gripper:
        state_mode = 'real_gripper'
    else:
        state_mode = 'plain'

    if state_dim == 7 and state_mode == 'plain':
        raise RuntimeError(
            "模型的 observation.state 是 7 维，但未显式启用 7D 状态模式。"
            "请使用 --state-gripper-placeholder 或 --state-with-gripper。"
        )
    if state_dim == 6 and state_mode != 'plain':
        raise RuntimeError(
            "模型的 observation.state 是 6 维，但启用了 7D 状态模式。"
            "请关闭 --state-gripper-placeholder / --state-with-gripper。"
        )

    print(f"      State dim: {state_dim} (mode={state_mode})")
    print(f"      Action dim: {action_dim if action_dim is not None else 'unknown'}")

    # 2. 初始化硬件
    print("\n[2/5] Initializing hardware...")
    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.arm_ip, args.arm_port)
    if handle.id == -1:
        raise RuntimeError(f"机械臂连接失败: {args.arm_ip}:{args.arm_port}")

    arm.rm_stop_drag_teach()
    arm.rm_set_tool_voltage(3)
    arm.rm_set_modbus_mode(GRIPPER_MODBUS_PORT, 9600, 2)
    time.sleep(0.3)

    gripper_params = rm_peripheral_read_write_params_t(
        port=GRIPPER_MODBUS_PORT, address=GRIPPER_MODBUS_ADDR,
        device=GRIPPER_MODBUS_DEVICE, num=GRIPPER_MODBUS_NUM
    )
    robot = RobotController(arm, gripper_params)

    cam_high_cfg = (calib_cfg.get("cameras") or {}).get("cam_high", {})
    cam_wrist_cfg = (calib_cfg.get("cameras") or {}).get("cam_wrist", {})
    exo_intrinsics = intrinsics_from_config(cam_high_cfg)
    ego_intrinsics = intrinsics_from_config(cam_wrist_cfg)
    T_base_exo = transform_from_config(cam_high_cfg, "T_base_cam")
    T_ee_ego = transform_from_config(cam_wrist_cfg, "T_ee_cam")
    target_locator = TargetLocator((runtime_cfg.get("target_locator") or {}))
    egexo_debugger = EgExoDebugger(runtime_cfg, enabled=args.debug_egexo)

    need_top_camera = use_cam_high or enable_egexo
    need_wrist_camera = use_cam_wrist or enable_egexo
    cam_top = (
        RealSenseCamera(
            args.cam_top,
            width=int(cam_high_cfg.get("width", 640)),
            height=int(cam_high_cfg.get("height", 480)),
            fps=max(1, int(round(args.freq))),
            enable_depth=enable_egexo,
        )
        if need_top_camera else None
    )
    cam_wrist = (
        DS87RosCamera(
            topic=args.cam_wrist_topic,
            width=int(cam_wrist_cfg.get("width", 640)),
            height=int(cam_wrist_cfg.get("height", 480)),
        )
        if need_wrist_camera else None
    )
    time.sleep(1)

    # # 3. 移动到初始位姿
    # print("\n[3/5] Moving to initial pose...")
    # robot.move_to_init(INIT_POSE)

    # 4. 等待确认
    print("\n[4/5] Ready to execute")
    if not args.headless:
        print("      按 Enter 开始推理...")
        input()
    else:
        print("      3秒后开始...")
        time.sleep(3)

    # 5. 推理循环
    print(f"\n[5/5] Executing {policy_type} at {args.freq}Hz (Ctrl+C to stop)...")
    step_count = 0
    control_period = 1.0 / args.freq
    policy.reset()

    try:
        while True:
            start_time = time.time()

            # 获取观测
            qpos = robot.get_qpos(include_gripper=False)
            if args.state_gripper_placeholder:
                qpos = np.concatenate([
                    qpos,
                    np.array([args.gripper_placeholder_value], dtype=np.float32)
                ])
            elif args.state_with_gripper:
                qpos = robot.get_qpos(include_gripper=True)
            observation = {
                'observation.state': torch.from_numpy(qpos).float(),
            }
            img_top_bgr = cam_top.get_frame() if cam_top else None
            depth_top = cam_top.get_depth_frame() if cam_top and enable_egexo else None
            img_wrist_bgr = cam_wrist.get_frame() if cam_wrist else None

            if use_cam_high and img_top_bgr is not None:
                img = cv2.cvtColor(img_top_bgr, cv2.COLOR_BGR2RGB)
                observation['observation.images.cam_high'] = \
                    torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            if use_cam_wrist and img_wrist_bgr is not None:
                img = cv2.cvtColor(img_wrist_bgr, cv2.COLOR_BGR2RGB)
                observation['observation.images.cam_wrist'] = \
                    torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            grounding = None
            phase = 0.0
            exo_roi = np.zeros(4, dtype=np.float32)
            if enable_egexo and img_top_bgr is not None and img_wrist_bgr is not None and depth_top is not None:
                ee_pose = robot.get_ee_pose()
                locator_result = target_locator.locate(img_top_bgr, depth=depth_top)
                exo_roi = np.asarray(locator_result.get("roi_xyxy", np.zeros(4, dtype=np.float32)), dtype=np.float32)

                if locator_result.get("valid", False):
                    grounding = project_exo_roi_to_ego(
                        exo_roi_xyxy=exo_roi,
                        exo_depth=depth_top,
                        exo_intrinsics=exo_intrinsics if exo_intrinsics.get("fx", 0.0) != 0.0 else cam_top.get_intrinsics(),
                        ego_intrinsics=ego_intrinsics,
                        T_base_exo=T_base_exo,
                        T_base_ee=ee_pose_to_T_base_ee(ee_pose),
                        T_ee_ego=T_ee_ego,
                        image_size=img_wrist_bgr.shape[:2],
                        cfg=calib_cfg.get("geometry", {}),
                    )
                else:
                    grounding = {
                        "valid": False,
                        "reason": locator_result.get("reason", "locator_invalid"),
                        "ego_roi_xyxy": np.zeros(4, dtype=np.float32),
                        "target_3d_base": np.zeros(3, dtype=np.float32),
                    }

                if not grounding.get("valid", False):
                    grounding["ego_roi_xyxy"] = np.zeros(4, dtype=np.float32)
                    grounding["target_3d_base"] = np.zeros(3, dtype=np.float32)
                phase = estimate_phase(ee_pose, grounding["target_3d_base"], grounding["valid"], runtime_cfg)

                if attach_egexo_features:
                    observation["observation.ee_pose"] = torch.from_numpy(ee_pose.astype(np.float32)).float()
                    observation["observation.grounding.ego_roi"] = torch.from_numpy(
                        np.asarray(grounding["ego_roi_xyxy"], dtype=np.float32)
                    ).float()
                    observation["observation.grounding.valid"] = torch.tensor(
                        [float(grounding["valid"])], dtype=torch.float32
                    )
                    observation["observation.target_3d_base"] = torch.from_numpy(
                        np.asarray(grounding["target_3d_base"], dtype=np.float32)
                    ).float()
                    observation["observation.phase"] = torch.tensor([phase], dtype=torch.float32)

                if args.debug_egexo:
                    egexo_debugger.log(
                        step=step_count,
                        grounding=grounding,
                        phase=phase,
                        exo_roi=exo_roi,
                        exo_image_bgr=img_top_bgr,
                        ego_image_bgr=img_wrist_bgr,
                    )

            # VLA 需要 language instruction
            if policy_type in ("pi0", "pi05", "smolvla", "egexo_smolvla"):
                observation['task'] = args.task

            # 预处理 → 推理 → 后处理
            observation = preprocessor(observation)

            with torch.no_grad():
                action_tensor = policy.select_action(observation)

            action_dict = postprocessor({'action': action_tensor})
            action = action_dict['action'][0].cpu().numpy().reshape(-1)
            if action.size < 6:
                raise RuntimeError(f"模型输出 action 维度不足 6D，当前为 {action.size}D")
            if action_dim is not None and action.size != action_dim:
                raise RuntimeError(
                    f"模型配置 action 维度为 {action_dim}D，但当前输出为 {action.size}D"
                )
            if action.size == 7 and not args.execute_gripper and step_count == 0:
                print("      注意: 模型输出 7D action，但未启用 --execute-gripper，第7维夹爪动作不会下发")

            # 执行
            robot.set_qpos(action, execute_gripper=args.execute_gripper)

            # 日志
            elapsed = time.time() - start_time
            if step_count % 10 == 0:
                actual_freq = 1.0 / elapsed if elapsed > 0 else 0
                gripper_state = "N/A"
                if len(action) > 6:
                    if args.execute_gripper:
                        gripper_state = "闭合" if action[6] > 0.5 else "打开"
                    else:
                        gripper_state = "已禁用"
                grounding_text = ""
                if enable_egexo and grounding is not None:
                    grounding_text = (
                        f" | G:{int(bool(grounding.get('valid', False)))}"
                        f" P:{phase:.1f}"
                    )
                print(f"[{policy_type}] Step {step_count:4d} | "
                      f"J1:{qpos[0]:6.1f}→{action[0]:6.1f} | "
                      f"夹爪:{gripper_state}{grounding_text} | {actual_freq:.1f}Hz")

            # 可视化
            if not args.headless:
                try:
                    frames = []
                    if img_top_bgr is not None:
                        top_vis = img_top_bgr.copy()
                        if enable_egexo:
                            roi_draw = np.asarray(exo_roi, dtype=np.int32).reshape(4)
                            if roi_draw[2] > roi_draw[0] and roi_draw[3] > roi_draw[1]:
                                cv2.rectangle(top_vis, (roi_draw[0], roi_draw[1]), (roi_draw[2], roi_draw[3]), (0, 255, 0), 2)
                        frames.append(cv2.resize(top_vis, (320, 240)))
                    if img_wrist_bgr is not None:
                        wrist_vis = img_wrist_bgr.copy()
                        if enable_egexo and grounding is not None:
                            ego_roi = np.asarray(grounding.get("ego_roi_xyxy", np.zeros(4)), dtype=np.int32).reshape(4)
                            if ego_roi[2] > ego_roi[0] and ego_roi[3] > ego_roi[1]:
                                cv2.rectangle(wrist_vis, (ego_roi[0], ego_roi[1]), (ego_roi[2], ego_roi[3]), (0, 255, 0), 2)
                        frames.append(cv2.resize(wrist_vis, (320, 240)))
                    if frames:
                        display = np.hstack(frames) if len(frames) > 1 else frames[0]
                        cv2.putText(display, f"{policy_type} Step: {step_count}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.imshow("Inference", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                except cv2.error:
                    args.headless = True

            step_count += 1

            # 频率控制
            remaining = control_period - (time.time() - start_time)
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\n\n推理终止")
    finally:
        robot.stop()
        try:
            if not args.headless:
                cv2.destroyAllWindows()
        except Exception:
            pass
        robot.close()
        if cam_top:
            cam_top.close()
        if cam_wrist:
            cam_wrist.close()
        print("硬件已关闭")


if __name__ == "__main__":
    main()
