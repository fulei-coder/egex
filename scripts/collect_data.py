#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 运行前请先执行以下命令：
"""
终端1 (启动相机 ROS 2 节点)
source /opt/ros/humble/setup.bash
source ~/ScepterSDK/3rd-PartyPlugin/ROS2/install/setup.bash
ros2 run ScepterROS scepter_camera

终端2 (运行数据采集脚本)
conda activate lerobot
cd ~/lerobot-realman-vla
python3 scripts/collect_data.py  --teaching
# 如果原始 hdf5 存到E盘
# python3 scripts/collect_data.py --teaching --save-dir /media//1252BAD252BABA35/raw_hdf5/pick_cube
"""

"""
数据采集脚本 — RealMan RM65 + Vive Tracker + 双相机

- 顶部相机：D435，本地 pyrealsense2
- 腕部相机：DS87，官方 ROS Image 话题订阅（真 RGB）
- 不使用 cv_bridge，直接手工解析 sensor_msgs/Image，绕开 libffi / cv_bridge 冲突
- 两个相机统一 15 FPS


HDF5:
  - observations/qpos
  - observations/images/cam_high
  - observations/images/cam_wrist
  - action
  - timestamps
"""

import select
import tty
import termios
import time
import threading
import sys
import os

# 屏蔽由于容器或系统缺少字体库导致的 OpenCV(Qt) 警告弹字
os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false;qt.text.font.*=false"

import h5py
import numpy as np
import cv2
import argparse

# ============ 配置区域 ============
DEFAULT_ARM_IP = "192.168.1.18"
DEFAULT_ARM_PORT = 8080

DEFAULT_CAM_TOP_SERIAL = "346122070612"  # D435 
# DEFAULT_CAM_TOP_SERIAL = "108222250854"  # D455 
DEFAULT_TRACKER_SERIAL = "LHR-00000000"

DEFAULT_DS87_RGB_TOPIC = "/Scepter/color/image_raw"
DEFAULT_DS87_RGB_TRANSFORMED_TOPIC = "/Scepter/transformedColor/image_raw"

RM_SDK_PATH = "/home/tony/RM_API2/Python"
if RM_SDK_PATH not in sys.path:
    sys.path.append(RM_SDK_PATH)

ROBOT_INIT_POS = np.array([-0.218, 0.06, 0.357])
ROBOT_INIT_ORI = np.array([-3.126, 0.001, -0.015])
# ============ 配置区域结束 ============


def get_next_filename(save_dir, task_name):
    os.makedirs(save_dir, exist_ok=True)
    idx = 0
    while os.path.exists(os.path.join(save_dir, f"{task_name}_{idx}.hdf5")):
        idx += 1
    return os.path.join(save_dir, f"{task_name}_{idx}.hdf5")


# ============ RealSense 相机模块 (D435) ============
import pyrealsense2 as rs

class D435Camera:
    def __init__(self, serial_number, width=640, height=480, fps=15):
        self.serial_number = str(serial_number)
        self.width, self.height = width, height
        self.fps = fps
        self.target_shape = (height, width, 3)
        self.latest_color = np.zeros(self.target_shape, dtype=np.uint8)
        self.lock = threading.Lock()
        self.stopped = False
        self.is_active = False
        self.frame_count = 0
        self.last_frame_time = 0.0

        try:
            self.pipeline = rs.pipeline()
            self.config = rs.config()

            try:
                ctx = rs.context()
                for dev in ctx.query_devices():
                    if dev.get_info(rs.camera_info.serial_number) == self.serial_number:
                        dev.hardware_reset()
                        time.sleep(2)
            except Exception:
                pass

            if self.serial_number:
                self.config.enable_device(self.serial_number)
            self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            profile = self.pipeline.start(self.config)
            self.is_active = True

            try:
                color_sensor = profile.get_device().first_color_sensor()
                if color_sensor.supports(rs.option.enable_auto_exposure):
                    color_sensor.set_option(rs.option.enable_auto_exposure, 0)
                if color_sensor.supports(rs.option.exposure):
                    color_sensor.set_option(rs.option.exposure, 150)
            except Exception:
                pass

            self.thread = threading.Thread(target=self._update_loop, daemon=True)
            self.thread.start()
            print(f"[D435] started at {self.width}x{self.height} @ {self.fps}fps")

        except Exception as e:
            print(f"[!] D435 Camera {self.serial_number}: {e}")
            self.is_active = False

    def _update_loop(self):
        while not self.stopped and self.is_active:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=2000)
                color_frame = frames.get_color_frame()
                if color_frame:
                    frame_data = np.asanyarray(color_frame.get_data())
                    if frame_data.shape == self.target_shape:
                        with self.lock:
                            self.latest_color = frame_data.copy()
                            self.frame_count += 1
                            self.last_frame_time = time.time()
            except Exception:
                pass

    def get_status(self):
        with self.lock:
            return {
                "is_active": self.is_active,
                "frame_count": self.frame_count,
                "last_frame_time": self.last_frame_time,
                "age_sec": time.time() - self.last_frame_time if self.last_frame_time > 0 else 999.0
            }

    def get_frame(self):
        with self.lock:
            return self.latest_color.copy()

    def close(self):
        self.stopped = True
        if self.is_active:
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
        age = time.time() - self.last_frame_time if self.last_frame_time > 0 else 999.0
        return {
            "is_active": self.is_active,
            "frame_count": self.frame_count,
            "last_frame_time": self.last_frame_time,
            "age_sec": age,
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


# ============ Vive 遥控模块 ============
class ViveController:
    def __init__(self, arm, arm_lock, tracker_serial=None, enable_vive=True):
        self.arm = arm
        self.arm_lock = arm_lock
        self.tracker_serial = tracker_serial or DEFAULT_TRACKER_SERIAL

        self.robot_init_pos = ROBOT_INIT_POS.copy()
        self.robot_init_ori = ROBOT_INIT_ORI.copy()

        self.vive_init_pos = None
        self.vive_init_ori = None
        self.control_enabled = False
        self.running = True
        self.vive = None
        self.tracker = None
        self.enable_vive = enable_vive

        if self.enable_vive:
            self._init_vive()
            self.thread = threading.Thread(target=self._control_loop, daemon=True)
            self.thread.start()
        else:
            print("示教模式: 不使用 Vive 遥控，手动移动机械臂")

    def _init_vive(self):
        try:
            sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'hardware'))
            import vive_tracker as triad_lib
            self.vive = triad_lib.triad_openvr()
            for name, device in self.vive.devices.items():
                if "tracker" in name.lower():
                    serial = device.get_serial()
                    if serial == self.tracker_serial:
                        self.tracker = device
                        break
            if self.tracker is None:
                for name, device in self.vive.devices.items():
                    if "tracker" in name.lower():
                        self.tracker = device
                        break
        except Exception as e:
            print(f"[!] Vive: {e}")

    def calibrate(self):
        if self.tracker is None:
            print("[!] Vive 未连接")
            return False

        print("校准中... 保持 Tracker 静止")
        poses = []
        for _ in range(30):
            euler = self.tracker.get_pose_euler()
            if euler:
                poses.append(euler)
            time.sleep(0.033)

        if not poses:
            print("[!] 校准失败")
            return False

        avg_pose = np.mean(poses, axis=0)
        self.vive_init_pos = np.array([avg_pose[0], avg_pose[1], avg_pose[2]])
        self.vive_init_ori = np.array([avg_pose[3], avg_pose[4], avg_pose[5]])
        print("校准完成")
        return True

    def enable(self):
        if self.vive_init_pos is None:
            print("请先校准(c)")
            return
        self.control_enabled = True
        print("遥控已启用")

    def disable(self):
        self.control_enabled = False
        print("遥控已暂停")

    def _control_loop(self):
        interval = 0.05
        while self.running:
            time.sleep(interval)
            if not self.control_enabled or self.tracker is None or self.vive_init_pos is None:
                continue

            try:
                euler = self.tracker.get_pose_euler()
                if euler is None:
                    continue

                cur_pos = np.array([euler[0], euler[1], euler[2]])
                cur_ori = np.array([euler[3], euler[4], euler[5]])

                delta_pos = cur_pos - self.vive_init_pos
                delta_ori = cur_ori - self.vive_init_ori

                scale_pos = 0.5
                scale_ori = 0.3

                target_pos = self.robot_init_pos + np.array([
                    -delta_pos[2] * scale_pos,
                    -delta_pos[0] * scale_pos,
                    delta_pos[1] * scale_pos
                ])

                target_ori = self.robot_init_ori + np.array([
                    -delta_ori[2] * scale_ori * np.pi / 180,
                    -delta_ori[1] * scale_ori * np.pi / 180,
                    delta_ori[0] * scale_ori * np.pi / 180
                ])

                target_pos[0] = np.clip(target_pos[0], -0.5, 0.1)
                target_pos[1] = np.clip(target_pos[1], -0.3, 0.3)
                target_pos[2] = np.clip(target_pos[2], 0.1, 0.6)

                target_6d = [
                    float(target_pos[0]), float(target_pos[1]), float(target_pos[2]),
                    float(target_ori[0]), float(target_ori[1]), float(target_ori[2])
                ]

                with self.arm_lock:
                    self.arm.rm_movep_canfd(target_6d, False, 0, 60)

            except Exception:
                pass

    def shutdown(self):
        self.running = False


# ============ 数据录制模块 ============
class DataRecorder:
    def __init__(self, arm, arm_lock, cam_top, cam_wrist, target_fps=15):
        self.arm = arm
        self.arm_lock = arm_lock
        self.cam_top = cam_top
        self.cam_wrist = cam_wrist
        self.is_recording = False
        self.filename = None
        self.target_fps = target_fps
        self.data_buffer = {'qpos': [], 'images_top': [], 'images_wrist': [], 'timestamps': []}
        self.dark_wrist_frames = 0

    def start(self, filename):
        if self.is_recording:
            return
        self.filename = filename
        self.is_recording = True
        self.dark_wrist_frames = 0
        self.data_buffer = {'qpos': [], 'images_top': [], 'images_wrist': [], 'timestamps': []}
        self.record_start_time = time.time()
        self.thread = threading.Thread(target=self._record_loop, daemon=True)
        self.thread.start()
        print(f"录制开始: {os.path.basename(filename)} (目标 {self.target_fps}Hz)")

    def stop(self):
        if not self.is_recording:
            return
        self.is_recording = False
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

                with self.arm_lock:
                    code, state = self.arm.rm_get_current_arm_state()

                joint_angles = state['joint'] if code == 0 else [0] * 6

                img_top = self.cam_top.get_frame()
                img_wrist = self.cam_wrist.get_frame()

                wrist_mean = float(img_wrist.mean())
                if wrist_mean < 3.0:
                    self.dark_wrist_frames += 1
                    if self.dark_wrist_frames <= 5 or self.dark_wrist_frames % 30 == 0:
                        print(
                            f"[WARN] recorder got dark wrist frame: "
                            f"mean={wrist_mean:.2f}, dark_count={self.dark_wrist_frames}"
                        )

                self.data_buffer['qpos'].append(joint_angles)
                self.data_buffer['images_top'].append(img_top)
                self.data_buffer['images_wrist'].append(img_wrist)
                self.data_buffer['timestamps'].append(timestamp)

            except Exception as e:
                print(f" >> Record error: {e}")

    def _save(self):
        if not self.data_buffer['qpos']:
            print(" >> 无数据")
            return

        qpos = self.data_buffer['qpos']
        actions = qpos[1:] + [qpos[-1]]
        timestamps = self.data_buffer['timestamps']

        if len(timestamps) > 1:
            intervals = np.diff(timestamps)
            actual_fps = 1.0 / np.mean(intervals)
            print(f"  实际帧率: {actual_fps:.1f} Hz (目标: {self.target_fps} Hz)")

        print(f"  腕部暗帧数: {self.dark_wrist_frames}/{len(qpos)}")

        try:
            with h5py.File(self.filename, 'w') as f:
                f.attrs['sim'] = False
                f.attrs['fps'] = self.target_fps
                f.attrs['wrist_dark_frames'] = self.dark_wrist_frames

                f.create_dataset('observations/qpos', data=np.array(qpos, dtype=np.float32))
                f.create_dataset('action', data=np.array(actions, dtype=np.float32))
                f.create_dataset('timestamps', data=np.array(timestamps, dtype=np.float64))
                f.create_dataset(
                    'observations/images/cam_high',
                    data=np.array(self.data_buffer['images_top'], dtype=np.uint8),
                    compression="gzip"
                )
                f.create_dataset(
                    'observations/images/cam_wrist',
                    data=np.array(self.data_buffer['images_wrist'], dtype=np.uint8),
                    compression="gzip"
                )
            print(f"保存: {os.path.basename(self.filename)} ({len(qpos)} frames)")
        except Exception as e:
            print(f"[!] 保存失败: {e}")


def main():
    parser = argparse.ArgumentParser(description='RealMan RML63 数据采集')
    parser.add_argument('--arm-ip', type=str, default=DEFAULT_ARM_IP, help='机械臂IP地址')
    parser.add_argument('--arm-port', type=int, default=DEFAULT_ARM_PORT, help='机械臂端口')
    parser.add_argument('--cam-top', type=str, default=DEFAULT_CAM_TOP_SERIAL, help='顶部相机序列号')
    parser.add_argument(
        '--ds87-topic',
        type=str,
        default=DEFAULT_DS87_RGB_TOPIC,
        help='DS87 ROS 图像话题，默认 /Scepter/color/image_raw，也可改为 /Scepter/transformedColor/image_raw'
    )
    parser.add_argument('--save-dir', type=str, default='data/raw_hdf5', help='数据保存目录')
    # parser.add_argument('--save-dir', type=str, default='data/raw_hdf5/pick_cube', help='数据保存目录')
    parser.add_argument('--task-name', type=str, default='task_pick_cube', help='任务名称')
    parser.add_argument('--fps', type=int, default=15, help='采集帧率，当前统一使用15')
    parser.add_argument('--teaching', action='store_true', help='示教模式（不用Vive）')
    args = parser.parse_args()

    if args.fps != 15:
        print(f"[WARN] 当前强制统一使用 15fps，忽略传入值 {args.fps}")
    args.fps = 15

    try:
        from Robotic_Arm.rm_robot_interface import RoboticArm, rm_thread_mode_e
    except ImportError:
        print("[!] 无法导入 Robotic_Arm SDK，请检查路径设置。")
        sys.exit(1)

    print("=" * 50)
    print(f"  RealMan RM65 {'示教' if args.teaching else 'Vive遥操作'} 数据采集")
    print("=" * 50)

    print("初始化顶部相机 (d435)...")
    cam_top = D435Camera(args.cam_top, fps=args.fps)

    print("初始化腕部相机 (ds87 ros rgb)...")
    cam_wrist = DS87RosCamera(topic=args.ds87_topic)

    time.sleep(2.0)

    print(f"相机状态: Top={'OK' if cam_top.is_active else 'FAIL'}, Wrist={'OK' if cam_wrist.is_active else 'FAIL'}")
    if hasattr(cam_wrist, "get_status"):
        status = cam_wrist.get_status()
        print(
            "[DS87-ROS] status after init: "
            f"active={status['is_active']}, "
            f"frame_count={status['frame_count']}, "
            f"last_frame_age={status['last_frame_age']}, "
            f"topic={status['topic']}"
        )

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

    vive_ctrl = ViveController(arm, arm_lock, enable_vive=not args.teaching)
    if not args.teaching:
        print(f"Vive: {'OK' if vive_ctrl.tracker else 'FAIL'}")

    recorder = DataRecorder(arm, arm_lock, cam_top, cam_wrist, args.fps)

    print("\n" + "-" * 50)
    if args.teaching:
        print("单键命令: s=录制开始  d=停止并保存  q=退出")
    else:
        print("单键命令: c=校准  w=遥控  e=停止遥控  s=录制开始  d=停止并保存  q=退出")
    print("建议流程: c -> w -> s -> 示教 -> d")
    print("-" * 50)

    old_settings = termios.tcgetattr(sys.stdin)
    cmd_buffer = ""

    try:
        tty.setcbreak(sys.stdin.fileno())
        print("> ", end='', flush=True)

        while True:
            # 实时显示两个相机的图像
            img_t = cam_top.get_frame()
            img_w = cam_wrist.get_frame()
            
            # 判断两张图均已成功获取
            if img_t is not None and img_w is not None and img_t.size > 0 and img_w.size > 0:
                h_t, w_t = img_t.shape[:2]
                h_w, w_w = img_w.shape[:2]
                
                # 如果两张图像高度不一致，先缩放其中的一张以进行平滑的水平拼接
                if h_t != h_w:
                    new_w_w = int((h_t / h_w) * w_w)
                    img_w_resized = cv2.resize(img_w, (new_w_w, h_t))
                    combined_img = cv2.hconcat([img_t, img_w_resized])
                else:
                    combined_img = cv2.hconcat([img_t, img_w])
                    
                # 显示图像弹窗（默认在后台刷取键盘消息1毫秒）
                cv2.imshow("Cameras Preview", combined_img)
            cv2.waitKey(1)

            # 依旧保留原本的 select 机制处理终端下的标准输入字符
            rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
            if not rlist:
                continue

            char = sys.stdin.read(1)

            if char in ['\n', '\r']:
                cmd = cmd_buffer.strip()
                cmd_buffer = ""

                print()

                if cmd == 'q':
                    break
                elif cmd == 'c' and not args.teaching:
                    vive_ctrl.calibrate()
                elif cmd == 'w' and not args.teaching:
                    vive_ctrl.enable()
                elif cmd == 'e' and not args.teaching:
                    vive_ctrl.disable()
                elif cmd == 's':
                    if hasattr(cam_wrist, "get_status"):
                        status = cam_wrist.get_status()
                        print(
                            "[DS87-ROS] status before record: "
                            f"active={status['is_active']}, "
                            f"frame_count={status['frame_count']}, "
                            f"last_frame_age={status['last_frame_age']}, "
                            f"topic={status['topic']}"
                        )
                    filename = get_next_filename(args.save_dir, args.task_name)
                    recorder.start(filename)
                elif cmd == 'd':
                    recorder.stop()
                    if hasattr(cam_wrist, "get_status"):
                        status = cam_wrist.get_status()
                        print(
                            "[DS87-ROS] status after record: "
                            f"active={status['is_active']}, "
                            f"frame_count={status['frame_count']}, "
                            f"last_frame_age={status['last_frame_age']}, "
                            f"topic={status['topic']}"
                        )
                elif cmd:
                    print(f"未知命令: {repr(cmd)}")

                print("> ", end='', flush=True)

            elif char == '\x7f':  # Backspace
                if cmd_buffer:
                    cmd_buffer = cmd_buffer[:-1]
                    sys.stdout.write('\b \b')
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
        vive_ctrl.shutdown()
        cam_top.close()
        cam_wrist.close()
        arm.rm_delete_robot_arm()
        cv2.destroyAllWindows()  # 新增销毁 OpenCV 窗口
        print("已退出")

if __name__ == '__main__':
    main()