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
      --arm-ip <YOUR_ARM_IP> --freq 30

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
from pathlib import Path

# ============ 默认硬件配置（根据你的硬件修改） ============
DEFAULT_ARM_IP = "<YOUR_ARM_IP>"           # 出厂默认 192.168.2.18
DEFAULT_ARM_PORT = 8080
DEFAULT_CAM_TOP_SERIAL = "<YOUR_TOP_CAM_SN>"     # rs-enumerate-devices | grep Serial
DEFAULT_CAM_WRIST_SERIAL = "<YOUR_WRIST_CAM_SN>"

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


# ============ RealSense 相机 ============
import pyrealsense2 as rs


class RealSenseCamera:
    """RealSense 相机异步采集"""

    def __init__(self, serial_number, width=640, height=480, fps=30):
        self.serial_number = str(serial_number)
        self.width, self.height = width, height
        self.latest_color = np.zeros((height, width, 3), dtype=np.uint8)
        self.lock = threading.Lock()
        self.stopped = False

        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(self.serial_number)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        try:
            profile = self.pipeline.start(self.config)
            color_sensor = profile.get_device().first_color_sensor()
            if color_sensor.supports(rs.option.enable_auto_exposure):
                color_sensor.set_option(rs.option.enable_auto_exposure, 0)
            if color_sensor.supports(rs.option.exposure):
                color_sensor.set_option(rs.option.exposure, 150)
            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print(f"[✓] Camera {self.serial_number} OK")
        except Exception as e:
            print(f"[✗] Camera {self.serial_number}: {e}")
            raise

    def _update_loop(self):
        while not self.stopped:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame_data = np.asanyarray(color_frame.get_data())
                    with self.lock:
                        self.latest_color = frame_data.copy()
            except Exception:
                pass

    def get_frame(self):
        with self.lock:
            return self.latest_color.copy()

    def close(self):
        self.stopped = True
        try:
            self.pipeline.stop()
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
        self.ema_alpha = 0.3       # EMA 系数 (0.3=平滑, 0.7=响应快)
        self.joint_deadzone = 0.5  # 死区阈值(度)

    def get_qpos(self, skip_gripper=True):
        """获取当前 [6关节角 + 夹爪位置]"""
        with self.lock:
            joint_state = self.arm.rm_get_current_arm_state()
            joint_angles = joint_state[1]['joint'][:6]

            if skip_gripper:
                gripper_pos = 0.5 if self._last_gripper_cmd is None else float(self._last_gripper_cmd)
            else:
                gripper_pos = self._cached_gripper_pos
                try:
                    ret, gripper_reg = self.arm.rm_read_multiple_holding_registers(self.gripper_params)
                    if ret == 0 and gripper_reg:
                        gripper_pos = 1 - register_to_dec(gripper_reg)
                        self._cached_gripper_pos = gripper_pos
                except Exception:
                    pass

            return np.array(joint_angles + [gripper_pos], dtype=np.float32)

    def set_qpos(self, qpos, use_smoothing=True):
        """设置目标关节角 + 夹爪"""
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

            if should_send:
                self.arm.rm_movej(joint_target.tolist(), 50, 0, 0, 0)
                self._last_joint_cmd = joint_target.copy()

            # 异步夹爪
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
    parser.add_argument('--cam-wrist', type=str, default=DEFAULT_CAM_WRIST_SERIAL, help='腕部相机序列号')
    parser.add_argument('--freq', type=float, default=15.0,
                        help='控制频率(Hz)，应与训练数据fps一致')
    parser.add_argument('--task', type=str, default='pick up the cube',
                        help='VLA 任务描述 (Pi0/SmolVLA 需要)')
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

    preprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_preprocessor.json')
    postprocessor = DataProcessorPipeline.from_pretrained(
        args.model, config_filename='policy_postprocessor.json')

    # 检测输入特征
    input_features = policy.config.input_features
    use_cam_high = 'observation.images.cam_high' in input_features
    use_cam_wrist = 'observation.images.cam_wrist' in input_features
    print(f"      Input features: {list(input_features.keys())}")

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

    cam_top = RealSenseCamera(args.cam_top) if use_cam_high else None
    cam_wrist = RealSenseCamera(args.cam_wrist) if use_cam_wrist else None
    time.sleep(1)

    # 3. 移动到初始位姿
    print("\n[3/5] Moving to initial pose...")
    robot.move_to_init(INIT_POSE)

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
            qpos = robot.get_qpos(skip_gripper=True)
            observation = {
                'observation.state': torch.from_numpy(qpos).float(),
            }

            if cam_top:
                img = cv2.cvtColor(cam_top.get_frame(), cv2.COLOR_BGR2RGB)
                observation['observation.images.cam_high'] = \
                    torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            if cam_wrist:
                img = cv2.cvtColor(cam_wrist.get_frame(), cv2.COLOR_BGR2RGB)
                observation['observation.images.cam_wrist'] = \
                    torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

            # VLA 需要 language instruction
            if policy_type in ("pi0", "pi05", "smolvla"):
                observation['task'] = args.task

            # 预处理 → 推理 → 后处理
            observation = preprocessor(observation)

            with torch.no_grad():
                action_tensor = policy.select_action(observation)

            action_dict = postprocessor({'action': action_tensor})
            action = action_dict['action'][0].cpu().numpy()

            # 执行
            robot.set_qpos(action)

            # 日志
            elapsed = time.time() - start_time
            if step_count % 10 == 0:
                actual_freq = 1.0 / elapsed if elapsed > 0 else 0
                gripper_state = "闭合" if action[6] > 0.5 else "打开"
                print(f"[{policy_type}] Step {step_count:4d} | "
                      f"J1:{qpos[0]:6.1f}→{action[0]:6.1f} | "
                      f"夹爪:{gripper_state} | {actual_freq:.1f}Hz")

            # 可视化
            if not args.headless:
                try:
                    frames = []
                    if cam_top:
                        frames.append(cv2.resize(cam_top.get_frame(), (320, 240)))
                    if cam_wrist:
                        frames.append(cv2.resize(cam_wrist.get_frame(), (320, 240)))
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
