#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U-arm -> RealMan 数据采集（6DoF执行，7DoF接口预留）

当前调试模式：
- 机械臂仅执行前6个关节
- 第7维为夹爪占位接口（默认0.0）
- HDF5 统一保存 qpos/action 为 7 维，便于后续无缝升级二指夹爪
"""

import argparse
import os
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import cv2
import h5py
import numpy as np
import yaml

# Keep OpenCV console clean on systems missing font stacks.
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.text.font.*=false"

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(REPO_ROOT / "hardware") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "hardware"))

import collect_data as collect_common
from realman_teleop_controller import RealmanTeleopController, SharedCommandBuffer
from uarm_realman_mapper import UarmRealmanMapper
from uarm_ros2_subscriber import UarmLeaderSubscriber


def resolve_repo_path(path_str):
    p = Path(path_str)
    if p.is_file():
        return p.resolve()

    candidate = (REPO_ROOT / path_str).resolve()
    if candidate.is_file():
        return candidate

    raise FileNotFoundError(f"文件不存在: {path_str}")


def load_map_config(path_str):
    path = resolve_repo_path(path_str)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"映射配置格式错误: {path}")

    return cfg, path


def get_next_filename(save_dir, task_name):
    return collect_common.get_next_filename(save_dir, task_name)


def get_robot_qpos7(arm, arm_lock, gripper_placeholder=0.0):
    with arm_lock:
        code, state = arm.rm_get_current_arm_state()

    joints = state["joint"][:6] if code == 0 else [0.0] * 6
    qpos = np.zeros(7, dtype=np.float32)
    qpos[:6] = np.asarray(joints, dtype=np.float32)
    qpos[6] = float(gripper_placeholder)
    return qpos


class DataRecorder:
    def _reset_buffer(self):
        self.data_buffer = {
            "qpos": [],
            "action": [],
            "images_top": [],
            "depth_top": [],
            "images_wrist": [],
            "ee_pose": [],
            "timestamps": [],
            "target_roi_exo": [],
            "target_3d_base": [],
            "grounding_valid": [],
            "leader_timestamp": [],
            "leader_age_sec": [],
            "controller_enabled": [],
            "controller_last_error": [],
            "cam_high_timestamp": [],
            "cam_high_age_sec": [],
            "cam_wrist_timestamp": [],
            "cam_wrist_age_sec": [],
            "cam_high_mean": [],
            "cam_wrist_mean": [],
            "movej_ret_code": [],
        }

    def __init__(
        self,
        arm,
        arm_lock,
        cam_top,
        cam_wrist,
        command_buffer,
        controller,
        target_fps=15,
        gripper_placeholder=0.0,
        active_dof=6,
        recording_cfg=None,
        calibration_cfg=None,
    ):
        self.arm = arm
        self.arm_lock = arm_lock
        self.cam_top = cam_top
        self.cam_wrist = cam_wrist
        self.command_buffer = command_buffer
        self.controller = controller
        self.target_fps = target_fps
        self.gripper_placeholder = float(gripper_placeholder)
        self.active_dof = int(active_dof)
        self.recording_cfg = recording_cfg or {}
        self.calibration_cfg = calibration_cfg or {}
        self.cam_high_meta = collect_common.get_camera_metadata(
            self.calibration_cfg,
            "cam_high",
            fallback_intrinsics=self.cam_top.get_intrinsics() if hasattr(self.cam_top, "get_intrinsics") else None,
        )
        self.cam_wrist_meta = collect_common.get_camera_metadata(
            self.calibration_cfg,
            "cam_wrist",
            fallback_intrinsics=None,
        )

        self.is_recording = False
        self.filename = None
        self.thread = None
        self.dark_wrist_frames = 0
        self._reset_buffer()

    def start(self, filename):
        if self.is_recording:
            return

        max_age = self.recording_cfg.get("max_camera_age_sec", 0.5)
        min_mean = self.recording_cfg.get("min_image_mean", 3.0)

        if not self.cam_top.is_active:
            print("[!] 顶部相机 (D435) 未激活，拒绝录制")
            return False
            
        top_status = self.cam_top.get_status() if hasattr(self.cam_top, "get_status") else {"age_sec": 0.0}
        if top_status.get("age_sec", 0.0) > max_age:
            print(f"[!] 顶部相机帧超时 ({top_status['age_sec']:.2f}s > {max_age}s)，拒绝录制")
            return False
            
        img_top = self.cam_top.get_frame()
        if img_top is None or float(img_top.mean()) < min_mean:
            print("[!] 顶部相机图像全黑或均值过低，拒绝录制")
            return False

        if not self.cam_wrist.is_active:
            print("[!] 腕部相机 (DS87) 未激活，拒绝录制")
            return False
            
        wrist_status = self.cam_wrist.get_status()
        if wrist_status.get("frame_count", 0) <= 0:
            print("[!] 腕部相机无有效帧 (frame_count=0)，拒绝录制")
            return False
            
        if wrist_status.get("age_sec", 0.0) > max_age:
            print(f"[!] 腕部相机帧超时 ({wrist_status['age_sec']:.2f}s > {max_age}s)，拒绝录制")
            return False
            
        img_wrist = self.cam_wrist.get_frame()
        if img_wrist is None or float(img_wrist.mean()) < min_mean:
            print("[!] 腕部相机图像全黑或均值过低，拒绝录制")
            return False

        self.filename = filename
        self.is_recording = True
        self.dark_wrist_frames = 0
        self.record_start_time = time.time()
        self._reset_buffer()

        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print(f"录制开始: {os.path.basename(filename)} (目标 {self.target_fps}Hz)")
        return True

    def stop(self):
        if not self.is_recording:
            return

        self.is_recording = False
        if self.thread is not None:
            self.thread.join()
        self._save()

    def _record_loop(self):
        interval = 1.0 / self.target_fps
        next_time = time.time()

        while self.is_recording:
            now = time.time()
            if now < next_time:
                time.sleep(0.001)
                continue
            next_time += interval

            try:
                timestamp = now - self.record_start_time

                qpos = get_robot_qpos7(
                    self.arm,
                    self.arm_lock,
                    gripper_placeholder=self.gripper_placeholder,
                )
                action, _ = self.command_buffer.get()

                # Keep interface stable: always 7D qpos/action.
                qpos = np.asarray(qpos, dtype=np.float32).reshape(-1)[:7]
                action = np.asarray(action, dtype=np.float32).reshape(-1)[:7]

                img_top = self.cam_top.get_frame()
                depth_top = self.cam_top.get_depth_frame() if hasattr(self.cam_top, "get_depth_frame") else None
                img_wrist = self.cam_wrist.get_frame()
                with self.arm_lock:
                    code, state = self.arm.rm_get_current_arm_state()
                ee_pose = collect_common.extract_ee_pose_from_realman_state(state if code == 0 else {})

                controller_status = self.controller.status()
                leader_record = self.controller.leader_subscriber.get()
                leader_ts = leader_record[1]
                leader_age = now - leader_ts

                top_status = self.cam_top.get_status() if hasattr(self.cam_top, "get_status") else {"age_sec": 0.0, "last_frame_time": 0.0}
                wrist_status = self.cam_wrist.get_status() if hasattr(self.cam_wrist, "get_status") else {"age_sec": 0.0, "last_frame_time": 0.0}

                wrist_mean = float(img_wrist.mean())
                if wrist_mean < 3.0:
                    self.dark_wrist_frames += 1
                    if self.dark_wrist_frames <= 5 or self.dark_wrist_frames % 30 == 0:
                        print(
                            "[WARN] recorder got dark wrist frame: "
                            f"mean={wrist_mean:.2f}, dark_count={self.dark_wrist_frames}"
                        )

                self.data_buffer["qpos"].append(qpos)
                self.data_buffer["action"].append(action)
                self.data_buffer["images_top"].append(img_top)
                self.data_buffer["depth_top"].append(
                    np.asarray(depth_top if depth_top is not None else np.zeros(img_top.shape[:2], dtype=np.uint16), dtype=np.uint16)
                )
                self.data_buffer["images_wrist"].append(img_wrist)
                self.data_buffer["ee_pose"].append(ee_pose.astype(np.float32))
                self.data_buffer["timestamps"].append(timestamp)
                self.data_buffer["target_roi_exo"].append(np.zeros(4, dtype=np.float32))
                self.data_buffer["target_3d_base"].append(np.zeros(3, dtype=np.float32))
                self.data_buffer["grounding_valid"].append(np.zeros(1, dtype=np.float32))
                self.data_buffer["leader_timestamp"].append(leader_ts)
                self.data_buffer["leader_age_sec"].append(leader_age)
                self.data_buffer["controller_enabled"].append(controller_status["enabled"])
                self.data_buffer["controller_last_error"].append(controller_status.get("last_error", ""))
                
                self.data_buffer["cam_high_timestamp"].append(top_status.get("last_frame_time", 0.0))
                self.data_buffer["cam_high_age_sec"].append(top_status.get("age_sec", 0.0))
                self.data_buffer["cam_wrist_timestamp"].append(wrist_status.get("last_frame_time", 0.0))
                self.data_buffer["cam_wrist_age_sec"].append(wrist_status.get("age_sec", 0.0))
                self.data_buffer["cam_high_mean"].append(float(img_top.mean()))
                self.data_buffer["cam_wrist_mean"].append(wrist_mean)
                self.data_buffer["movej_ret_code"].append(controller_status.get("last_movej_ret", 0))

            except Exception as e:
                print(f" >> Record error: {e}")

    def _save(self):
        if not self.data_buffer["qpos"]:
            print(" >> 无数据")
            return

        qpos = np.asarray(self.data_buffer["qpos"], dtype=np.float32)
        action = np.asarray(self.data_buffer["action"], dtype=np.float32)
        depth_top = np.asarray(self.data_buffer["depth_top"], dtype=np.uint16)
        ee_pose = np.asarray(self.data_buffer["ee_pose"], dtype=np.float32)
        timestamps = np.asarray(self.data_buffer["timestamps"], dtype=np.float64)
        target_roi_exo = np.asarray(self.data_buffer["target_roi_exo"], dtype=np.float32)
        target_3d_base = np.asarray(self.data_buffer["target_3d_base"], dtype=np.float32)
        grounding_valid = np.asarray(self.data_buffer["grounding_valid"], dtype=np.float32)
        
        leader_timestamp = np.asarray(self.data_buffer["leader_timestamp"], dtype=np.float64)
        leader_age_sec = np.asarray(self.data_buffer["leader_age_sec"], dtype=np.float32)
        controller_enabled = np.asarray(self.data_buffer["controller_enabled"], dtype=bool)
        
        cam_high_timestamp = np.asarray(self.data_buffer["cam_high_timestamp"], dtype=np.float64)
        cam_high_age_sec = np.asarray(self.data_buffer["cam_high_age_sec"], dtype=np.float32)
        cam_wrist_timestamp = np.asarray(self.data_buffer["cam_wrist_timestamp"], dtype=np.float64)
        cam_wrist_age_sec = np.asarray(self.data_buffer["cam_wrist_age_sec"], dtype=np.float32)
        cam_high_mean = np.asarray(self.data_buffer["cam_high_mean"], dtype=np.float32)
        cam_wrist_mean = np.asarray(self.data_buffer["cam_wrist_mean"], dtype=np.float32)
        movej_ret_code = np.asarray(self.data_buffer["movej_ret_code"], dtype=np.int32)
        
        # Convert string errors to bytes for HDF5 compatibility
        controller_last_error = np.array([e.encode('utf-8') for e in self.data_buffer["controller_last_error"]], dtype=h5py.string_dtype(encoding='utf-8') if hasattr(h5py, 'string_dtype') else "S")

        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            actual_fps = 1.0 / np.mean(intervals)
            print(f"  实际帧率: {actual_fps:.1f} Hz (目标: {self.target_fps} Hz)")

        print(f"  腕部暗帧数: {self.dark_wrist_frames}/{len(qpos)}")

        try:
            with h5py.File(self.filename, "w") as f:
                f.attrs["sim"] = False
                f.attrs["fps"] = self.target_fps
                f.attrs["active_dof"] = self.active_dof
                f.attrs["wrist_dark_frames"] = self.dark_wrist_frames

                f.create_dataset("observations/qpos", data=qpos)
                f.create_dataset("action", data=action)
                f.create_dataset("timestamps", data=timestamps)
                f.create_dataset(
                    "observations/images/cam_high",
                    data=np.array(self.data_buffer["images_top"], dtype=np.uint8),
                    compression="gzip",
                )
                f.create_dataset(
                    "observations/depth/cam_high",
                    data=depth_top,
                    compression="gzip",
                )
                f.create_dataset(
                    "observations/images/cam_wrist",
                    data=np.array(self.data_buffer["images_wrist"], dtype=np.uint8),
                    compression="gzip",
                )
                f.create_dataset("observations/ee_pose", data=ee_pose)
                f.create_dataset("observations/grounding/target_roi_exo", data=target_roi_exo)
                f.create_dataset("observations/grounding/target_3d_base", data=target_3d_base)
                f.create_dataset("observations/grounding/valid", data=grounding_valid)
                
                f.create_dataset("metadata/leader_timestamp", data=leader_timestamp)
                f.create_dataset("metadata/leader_age_sec", data=leader_age_sec)
                f.create_dataset("metadata/controller_enabled", data=controller_enabled)
                f.create_dataset("metadata/controller_last_error", data=controller_last_error)
                f.create_dataset("metadata/movej_ret_code", data=movej_ret_code)
                f.create_dataset("metadata/cam_high_timestamp", data=cam_high_timestamp)
                f.create_dataset("metadata/cam_high_age_sec", data=cam_high_age_sec)
                f.create_dataset("metadata/cam_wrist_timestamp", data=cam_wrist_timestamp)
                f.create_dataset("metadata/cam_wrist_age_sec", data=cam_wrist_age_sec)
                f.create_dataset("metadata/cam_high_mean", data=cam_high_mean)
                f.create_dataset("metadata/cam_wrist_mean", data=cam_wrist_mean)
                f.create_dataset("metadata/cameras/cam_high/intrinsics", data=self.cam_high_meta["intrinsics_array"])
                f.create_dataset("metadata/cameras/cam_high/dist_coeffs", data=self.cam_high_meta["dist_coeffs"])
                f.create_dataset("metadata/cameras/cam_high/T_base_cam", data=self.cam_high_meta["transform"])
                f.create_dataset("metadata/cameras/cam_wrist/intrinsics", data=self.cam_wrist_meta["intrinsics_array"])
                f.create_dataset("metadata/cameras/cam_wrist/dist_coeffs", data=self.cam_wrist_meta["dist_coeffs"])
                f.create_dataset("metadata/cameras/cam_wrist/T_ee_cam", data=self.cam_wrist_meta["transform"])

            print(f"保存: {os.path.basename(self.filename)} ({len(qpos)} frames)")
            print(f"  qpos shape={qpos.shape}, action shape={action.shape}")
            print(f"  depth shape={depth_top.shape}, ee_pose shape={ee_pose.shape}")
        except Exception as e:
            print(f"[!] 保存失败: {e}")


def main():
    parser = argparse.ArgumentParser(description="U-arm -> RealMan 数据采集（6DoF执行，7DoF预留）")
    parser.add_argument("--arm-ip", type=str, default=collect_common.DEFAULT_ARM_IP, help="机械臂IP地址")
    parser.add_argument("--arm-port", type=int, default=collect_common.DEFAULT_ARM_PORT, help="机械臂端口")
    parser.add_argument("--cam-top", type=str, default=collect_common.DEFAULT_CAM_TOP_SERIAL, help="顶部相机序列号")
    parser.add_argument("--ds87-topic", type=str, default=collect_common.DEFAULT_DS87_RGB_TOPIC, help="DS87 ROS图像话题")
    parser.add_argument("--save-dir", type=str, default="data/raw_hdf5", help="数据保存目录")
    parser.add_argument("--task-name", type=str, default="task_pick_cube", help="任务名称")
    parser.add_argument("--fps", type=int, default=15, help="采集帧率（当前固定15）")
    parser.add_argument("--map-config", type=str, default="configs/uarm_realman_map.yaml", help="映射配置文件")
    parser.add_argument("--uarm-topic", type=str, default="", help="覆盖配置中的 leader topic")
    parser.add_argument("--dry-run", action="store_true", help="仅计算映射不下发机械臂")
    parser.add_argument("--active-dof", type=int, default=None, help="覆盖配置中的 active_dof（默认6）")
    parser.add_argument("--gripper-enabled", action="store_true", help="启用第7维夹爪执行（默认关闭）")
    parser.add_argument("--movej-speed", type=float, default=None, help="覆盖配置中的 rm_movej 速度")
    parser.add_argument(
        "--calibration-config",
        type=str,
        default="configs/calibration_realman.yaml",
        help="相机与外参标定配置文件",
    )
    args = parser.parse_args()

    if args.fps != 15:
        print(f"[WARN] 当前强制统一使用 15fps，忽略传入值 {args.fps}")
    args.fps = 15

    map_cfg, cfg_path = load_map_config(args.map_config)

    if args.uarm_topic:
        map_cfg["leader_topic"] = args.uarm_topic

    if args.active_dof is not None:
        map_cfg["active_dof"] = int(args.active_dof)

    if args.movej_speed is not None:
        map_cfg.setdefault("control", {})["movej_speed"] = float(args.movej_speed)

    map_cfg.setdefault("gripper", {})
    if args.gripper_enabled:
        map_cfg["gripper"]["enabled"] = True

    calibration_cfg, calibration_path = collect_common.load_camera_calibration_config(args.calibration_config)

    # Stage-1 requirement: debug in 6DoF first, keep 7th dim as reserved interface.
    map_cfg["active_dof"] = int(max(1, min(6, map_cfg.get("active_dof", 6))))

    try:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    except ImportError:
        print("[!] 无法导入 Robotic_Arm SDK，请检查路径设置。")
        sys.exit(1)

    print("=" * 56)
    print("  U-arm -> RealMan 采集（6DoF执行，7DoF接口预留）")
    print("=" * 56)
    print(f"配置文件: {cfg_path}")
    print(f"标定文件: {calibration_path}")
    print(f"active_dof: {map_cfg['active_dof']}")
    print(f"gripper.enabled: {bool(map_cfg.get('gripper', {}).get('enabled', False))}")
    print(f"dry_run: {args.dry_run}")

    print("初始化顶部相机 (D435/D455 RGB-D)...")
    cam_top = collect_common.D435Camera(args.cam_top, fps=args.fps, enable_depth=True)

    print("初始化腕部相机 (DS87 ROS RGB)...")
    cam_wrist = collect_common.DS87RosCamera(topic=args.ds87_topic)

    time.sleep(2.0)
    print(f"相机状态: Top={'OK' if cam_top.is_active else 'FAIL'}, Wrist={'OK' if cam_wrist.is_active else 'FAIL'}")

    arm = RoboticArm(rm_thread_mode_e.RM_TRIPLE_MODE_E)
    handle = arm.rm_create_robot_arm(args.arm_ip, args.arm_port)
    if handle.id == -1:
        print("[!] 机械臂连接失败")
        cam_top.close()
        cam_wrist.close()
        sys.exit(1)

    print(f"机械臂: OK ({args.arm_ip}:{args.arm_port})")
    arm.rm_set_arm_run_mode(1)
    arm_lock = threading.Lock()

    collect_common.ensure_ros_node()
    ros_node = collect_common._ros_node

    leader_topic = map_cfg.get("leader_topic", "/servo_angles")
    leader_dim = int(map_cfg.get("leader_dim", 7))
    leader_sub = UarmLeaderSubscriber(ros_node, topic=leader_topic, leader_dim=leader_dim)

    mapper = UarmRealmanMapper(map_cfg)

    init_joints = np.asarray(map_cfg.get("robot_init_qpos_deg", [0, 0, 0, 0, 0, 0]), dtype=np.float32).reshape(-1)
    if init_joints.size < 6:
        init_joints = np.pad(init_joints, (0, 6 - init_joints.size), mode="constant", constant_values=0.0)

    init_action = np.zeros(7, dtype=np.float32)
    init_action[:6] = init_joints[:6]
    init_action[6] = mapper.get_gripper_placeholder()

    command_buffer = SharedCommandBuffer(dim=7, default_action=init_action)

    ctrl_cfg = map_cfg.get("control", {})
    controller = RealmanTeleopController(
        arm=arm,
        arm_lock=arm_lock,
        leader_subscriber=leader_sub,
        mapper=mapper,
        command_buffer=command_buffer,
        loop_hz=float(ctrl_cfg.get("loop_hz", 20)),
        movej_speed=float(ctrl_cfg.get("movej_speed", 5)),
        leader_timeout_sec=ctrl_cfg.get("leader_timeout_sec", 0.3),
        dry_run=args.dry_run,
        execute_gripper=bool(map_cfg.get("gripper", {}).get("enabled", False)),
        gripper_command_callback=None,
    )

    recorder = DataRecorder(
        arm=arm,
        arm_lock=arm_lock,
        cam_top=cam_top,
        cam_wrist=cam_wrist,
        command_buffer=command_buffer,
        controller=controller,
        target_fps=args.fps,
        gripper_placeholder=mapper.get_gripper_placeholder(),
        active_dof=map_cfg["active_dof"],
        recording_cfg=map_cfg.get("recording", {}),
        calibration_cfg=calibration_cfg,
    )

    print("\n" + "-" * 56)
    print("单键命令: c=校准  w=启用跟随  e=停止跟随  s=录制开始  d=停止并保存  q=退出")
    print("建议流程: c -> w -> s -> d")
    print("-" * 56)

    old_settings = termios.tcgetattr(sys.stdin)
    cmd_buffer = ""

    try:
        tty.setcbreak(sys.stdin.fileno())
        print("> ", end="", flush=True)

        while True:
            img_t = cam_top.get_frame()
            img_w = cam_wrist.get_frame()

            if img_t is not None and img_w is not None and img_t.size > 0 and img_w.size > 0:
                h_t, w_t = img_t.shape[:2]
                h_w, w_w = img_w.shape[:2]

                if h_t != h_w:
                    new_w_w = int((h_t / h_w) * w_w)
                    img_w_resized = cv2.resize(img_w, (new_w_w, h_t))
                    combined_img = cv2.hconcat([img_t, img_w_resized])
                else:
                    combined_img = cv2.hconcat([img_t, img_w])

                status = controller.status()
                state_text = "FOLLOW" if status["enabled"] else "IDLE"
                if args.dry_run:
                    state_text = f"{state_text} DRY-RUN"
                cv2.putText(
                    combined_img,
                    state_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if status["enabled"] else (0, 255, 255),
                    2,
                )
                cv2.imshow("Cameras Preview", combined_img)
            cv2.waitKey(1)

            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not rlist:
                continue

            char = sys.stdin.read(1)

            if char in ["\n", "\r"]:
                cmd = cmd_buffer.strip()
                cmd_buffer = ""
                print()

                if cmd == "q":
                    break
                elif cmd == "c":
                    robot_init_qpos = get_robot_qpos7(
                        arm,
                        arm_lock,
                        gripper_placeholder=mapper.get_gripper_placeholder(),
                    )
                    ok, msg = controller.calibrate(robot_init_qpos)
                    print(msg)
                    if ok:
                        command_buffer.set(robot_init_qpos)
                elif cmd == "w":
                    st = controller.status()
                    if not st["calibrated"]:
                        print("请先按 c 完成校准")
                    else:
                        controller.enable()
                        print("U-arm 跟随已启用")
                elif cmd == "e":
                    controller.disable()
                    print("U-arm 跟随已暂停")
                elif cmd == "s":
                    st = controller.status()
                    if not st["calibrated"]:
                        print("请先按 c 完成校准")
                    else:
                        filename = get_next_filename(args.save_dir, args.task_name)
                        recorder.start(filename)
                elif cmd == "d":
                    recorder.stop()
                elif cmd:
                    print(f"未知命令: {repr(cmd)}")

                print("> ", end="", flush=True)

            elif char == "\x7f":
                if cmd_buffer:
                    cmd_buffer = cmd_buffer[:-1]
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            else:
                cmd_buffer += char
                sys.stdout.write(char)
                sys.stdout.flush()

    except KeyboardInterrupt:
        print()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)

        if recorder.is_recording:
            recorder.stop()

        controller.shutdown()
        leader_sub.close()

        cam_top.close()
        cam_wrist.close()

        arm.rm_delete_robot_arm()
        cv2.destroyAllWindows()
        print("已退出")


if __name__ == "__main__":
    main()
