import serial
import time 
import numpy as np
import re
import gymnasium as gym
import mani_skill.envs  # 必须导入以注册所有 env/agent
from threading import Event, Thread, Lock
from queue import Queue, Empty
import torch
import sapien
import argparse
from mani_skill.utils import sapien_utils


class ServoTeleoperatorSim: 
    """机械臂遥操作仿真系统
    
    支持通过串口读取舵机角度，并映射到不同类型的机械臂仿真环境中。
    支持的机械臂类型：arx-x5, so100, xarm6_robotiq, panda, x_fetch, unitree_h1
    """
    
    def __init__(self, scene: str, robot_uids: str, serial_port: str = '/dev/ttyUSB0'):
        """初始化遥操作系统
        
        Args:
            scene: 仿真场景名称
            robot_uids: 机械臂类型标识符
            serial_port: 串口设备路径
        """
        # 串口配置
        self.SERIAL_PORT = serial_port
        self.BAUDRATE = 115200
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.01)

        # 系统配置
        self.scene = scene
        self.robot_uids = robot_uids
        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 7  # 舵机初始校准角度
        self.sim_init_angles = [0.0] * 7  # 仿真初始角度
        self.stop_event = Event()
        self.rate = 50.0  # 控制频率
        
        # 初始化舵机并校准零位
        self._init_servos()

        # 线程安全的数据交换队列
        self.arm_pos_queue: "Queue[list]" = Queue(maxsize=1)

        # 根据机器人类型选择控制模式
        if robot_uids == "x_fetch": 
            self.control_mode = "pd_joint_pos_dual_arm"
        elif robot_uids == "unitree_h1":
            self.control_mode = "pd_joint_pos"
        else: 
            self.control_mode = "pd_joint_pos"

        # 创建仿真环境
        self.env = gym.make(
            scene,
            robot_uids=robot_uids,
            render_mode="human",
            control_mode=self.control_mode,
            sensor_configs=dict(shader_pack="rt-fast"),
            human_render_camera_configs=dict(shader_pack="rt-fast"),
            viewer_camera_configs=dict(shader_pack="rt-fast"),
            sim_config=dict(
                default_materials_config=dict(
                    static_friction=10.0,  # 静摩擦
                    dynamic_friction=10.0, # 动摩擦
                    restitution=0.0       # 反弹系数
                )
            ),
        )
        obs, _ = self.env.reset(seed=0)
        print("Action space:", self.env.action_space)
        
        # 为H1设置初始站立姿态
        if robot_uids == "unitree_h1":
            self._setup_h1_standing_pose()

        # 创建生产者线程（读取舵机角度）
        self.produce_thread = Thread(
            target=self.angle_stream_loop, 
            args=(self.default_sender,), 
            daemon=True
        )

        # 创建消费者线程（控制仿真）
        self.consume_thread = Thread(
            target=self.pose_consumer_loop, 
            args=(self.teleop_sim_handler,), 
            daemon=True
        )

        self._setup_camera_pose()


    def _setup_camera_pose(self):
        agent = getattr(self.env.unwrapped, "agent", None)
        pose = sapien.Pose()
        if agent is not None:
            pose = agent.robot.get_pose()  # 返回 sapien.Pose
            print(f"机器人初始位置: {pose}")
        camera_pose = sapien_utils.look_at(
            [0.0, -1.5, 1.7], pose.p
        )
        camera_viewer = getattr(self.env.unwrapped, "viewer", None)
        if camera_viewer is not None:
            print(camera_pose)
            camera_pose_arr = camera_pose.raw_pose.squeeze().cpu().numpy()
            camera_position = camera_pose_arr[:3]
            camera_quaternion = camera_pose_arr[3:]
            camera_viewer.set_camera_pose(sapien.Pose(camera_position, camera_quaternion))

    def _setup_h1_standing_pose(self):
        """为H1机器人设置初始站立姿态"""
        try:
            agent = getattr(self.env.unwrapped, "agent", None)
            if agent is not None:
                # 使用H1预定义的站立姿态
                standing_keyframe = agent.keyframes["standing"]
                
                # 检查qpos的维度
                if hasattr(standing_keyframe.qpos, '__len__') and len(standing_keyframe.qpos) >= 19:
                    agent.reset(standing_keyframe.qpos)
                    agent.robot.set_root_pose(standing_keyframe.pose)
                    print("H1 已设置为站立姿态")
                else:
                    print("警告: standing_keyframe.qpos维度不正确，使用默认站立姿态")
                    # 使用默认的站立姿态
                    default_standing = np.array([
                        0, 0, 0, 0, 0, 0, 0, -0.4, -0.4, 0.0, 0.0, 0.8, 0.8, 0.0, 0.0, -0.4, -0.4, 0.0, 0.0
                    ])
                    agent.reset(default_standing)
                    agent.robot.set_root_pose(standing_keyframe.pose)
                    print("H1 已设置为默认站立姿态")
        except Exception as e:
            print(f"设置 H1 站立姿态失败: {e}")
    
    def _init_servos(self):
        """初始化舵机并校准零位角度"""
        self.send_command('#000PVER!')
        for i in range(7):
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip())
            self.zero_angles[i] = angle if angle is not None else 0.0
        print("[INFO] 舵机初始角度校准完成")

    def send_command(self, cmd: str) -> str:
        """发送串口命令并读取响应
        
        Args:
            cmd: 要发送的命令字符串
            
        Returns:
            响应字符串，如果无响应则返回空字符串
        """
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        response = self.ser.read_all()
        return response.decode('ascii', errors='ignore') if response else ""
    
    def pwm_to_angle(self, response_str: str, pwm_min: int = 500, 
                     pwm_max: int = 2500, angle_range: float = 270):
        """将PWM响应转换为角度
        
        Args:
            response_str: 舵机响应字符串
            pwm_min: PWM最小值
            pwm_max: PWM最大值
            angle_range: 角度范围（度）
            
        Returns:
            角度值，如果解析失败返回None
        """
        match = re.search(r'P(\d{4})', response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = pwm_max - pwm_min
        angle = (pwm_val - pwm_min) / pwm_span * angle_range
        return angle
    
    def publish_arm_pos(self, arm_pos: list):
        """发布最新的手臂位置到队列中，覆盖旧值"""
        try:
            # 清空队列中的旧数据
            while True:
                self.arm_pos_queue.get_nowait()
        except Empty:
            pass
        try:
            # 添加新数据
            self.arm_pos_queue.put_nowait(list(arm_pos))
        except Exception:
            pass
    
    def get_latest_arm_pos(self, timeout: float = 0.0):
        """获取最新的手臂位置快照
        
        Args:
            timeout: 超时时间，0表示立即返回
            
        Returns:
            最新的手臂位置列表，如果队列为空返回None
        """
        try:
            return self.arm_pos_queue.get(timeout=timeout) if timeout and timeout > 0 else self.arm_pos_queue.get_nowait()
        except Empty:
            return None
    
    def angle_to_gripper(self, angle_rad: float, pos_min: float, pos_max: float, 
                        angle_range: float = 1.5 * np.pi) -> float:
        """将舵机角度映射到夹爪位置
        
        Args:
            angle_rad: 舵机角度（弧度）
            pos_min: 夹爪最小位置
            pos_max: 夹爪最大位置
            angle_range: 舵机角度范围
            
        Returns:
            夹爪位置值
        """
        ratio = max(0, 1 - (angle_rad / angle_range))
        position = pos_min + (pos_max - pos_min) * ratio
        return float(np.clip(position, pos_min, pos_max))

    def convert_pose_to_action(self, pose: list) -> np.ndarray: 
        """根据不同机械臂类型将舵机位置转换为仿真动作
        
        Args:
            pose: 7维舵机角度列表（弧度）
            
        Returns:
            对应机械臂的动作向量
        """
        action = np.array([])

        if self.robot_uids == "arx-x5":  # 6轴机械臂 + 双指夹爪
            action = np.array(pose)
            # 处理夹爪：最后一维映射到夹爪位置
            action[-1] = self.angle_to_gripper(action[-1], 0, 0.044)
            action = np.concatenate([action, [action[-1]]])

            action[2] = -action[2]
            action[4], action[5] = -action[5], -action[4]  # 交换关节4和5
        
        elif self.robot_uids == "piper":  # 6轴机械臂 + 双指夹爪
            action = np.array(pose)
            action[-1] = self.angle_to_gripper(action[-1], 0, 0.04)

            action = np.concatenate([action, [action[-1]]])
            action[3], action[4] = action[4], -action[3]  # 交换关节4和5

        elif self.robot_uids == "so100":  # 5轴机械臂
            pose_copy = pose.copy()
            pose_copy.pop(5)  # 移除第6维（so100只有5轴）
            action = np.array(pose_copy)
            action[-1] = self.angle_to_gripper(action[-1], -1.1, 1.1)
            
            action[0] = -action[0]
            action[3] = -action[3]
            action[4] = -action[4]

        elif self.robot_uids == "xarm6_robotiq":  # 6轴机械臂 + Robotiq夹爪
            action = np.array(pose)
            action[3], action[4] = action[4], -action[3]  # 交换关节3和4
            # action[1] = -action[1]
            action[-1] = 0.81 - self.angle_to_gripper(action[-1], 0, 0.81)

        elif self.robot_uids == "panda":  # 7轴机械臂
            pose_copy = pose.copy()
            pose_copy.insert(2, 0.0)  # 在第3位插入0（Panda的第3关节）
            action = np.array(pose_copy)
            # action[1] = -action[1]
            action[3] = -action[3]
            action[4], action[5] = action[5], action[4]  # 交换关节4和5
            action[-1] = self.angle_to_gripper(action[-1], -1.0, 1.0)

        elif self.robot_uids == "x_fetch":  # 双臂机器人 + 移动底盘
            pose_copy = pose.copy()
            pose_copy.pop(5)  # 移除第6维
            action = np.array(pose_copy)
            action[-1] = self.angle_to_gripper(action[-1], -1.1, 1.1)
            # 调整关节方向
            action[0] = -action[0]
            action[1] = -action[1]
            action[3] = -action[3]
            action[4] = -action[4]
            # 构建双臂动作
            left_arm_action = action.copy()
            right_arm_action = left_arm_action.copy()
            right_arm_action[0] = -right_arm_action[0]
            right_arm_action[-2] = -right_arm_action[-2]
            # 组合：左臂关节 + 右臂关节 + 左右夹爪 + 底盘运动（0）
            zero_action = np.zeros(6)
            action = np.concatenate([
                left_arm_action[0:-1], 
                right_arm_action[0:-1], 
                [left_arm_action[-1], right_arm_action[-1]], 
                np.zeros(4) 
            ])

        elif self.robot_uids == "widowx250s":  # 6轴机械臂 + 双指夹爪
            action = np.array(pose)
            action[-1] = self.angle_to_gripper(action[-1], 0, 0.04)
            action = np.concatenate([action, [action[-1]]])
            action[3], action[4] = action[4], -action[3] 
        
        elif self.robot_uids == "unitree_h1":  # 人形机器人
            raw = np.array(pose, dtype=np.float32)
            action = np.zeros(19, dtype=np.float32)

            # 只修改手臂关节的增量（相对于当前状态）
            # 左臂：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            action[5] = raw[0]  # left_shoulder_pitch 增量
            action[9] = raw[1]   # left_shoulder_roll 增量
            action[13] = raw[2]  # left_shoulder_yaw 增量
            action[17] = raw[3]  # left_elbow 增量

            # 右臂：shoulder_pitch, shoulder_roll, shoulder_yaw, elbow
            action[6] = raw[4]   # right_shoulder_pitch 增量
            action[10] = raw[5]  # right_shoulder_roll 增量
            action[14] = raw[6]  # right_shoulder_yaw 增量
            action[18] = raw[3]  # right_elbow 增量（复用第4个舵机）

        else: 
            raise ValueError(f"不支持的机械臂类型: {self.robot_uids}")

        return action

    def default_sender(self, arm_pos: list): 
        """默认的角度发送回调（用于调试）"""
        print(f"舵机角度(度): {np.degrees(arm_pos)}")

    def teleop_sim_handler(self, action: np.ndarray, dwell: float = 0.01):
        """仿真控制处理函数
        
        Args:
            action: 机械臂动作向量
            dwell: 延迟时间
        """
        if self.env is None or action is None:
            return
            
        # 所有机器人类型都执行动作
        self.env.step(action)
        self.env.render()
        time.sleep(dwell)
    
    def angle_stream_loop(self, on_send):
        """角度数据生产者线程：周期性读取舵机角度
        
        Args:
            on_send: 回调函数，接收角度数据列表
        """
        num_joints = 7
        arm_pos = [0.0] * num_joints

        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()

        while not self.stop_event.is_set():
            # 读取所有关节角度
            for i in range(num_joints):
                response = self.send_command(f'#{i:03d}PRAD!')
                angle = self.pwm_to_angle(response.strip())
                if angle is not None:
                    # 计算相对于零位的角度
                    new_angle = angle - self.zero_angles[i]
                    arm_pos[i] = np.radians(new_angle)
                else: 
                    raise ValueError(f"舵机{i}回传异常: {response.strip()}")
            
            # 发布最新数据并调用回调
            self.publish_arm_pos(arm_pos)
            try:
                on_send(list(arm_pos))
            except Exception as e:
                print(f"角度发送回调异常: {e}")
                
            # 维持固定频率
            next_time += period
            sleep_dt = next_time - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                # 如果落后较多，重新同步时间
                next_time = time.monotonic()

    def pose_consumer_loop(self, on_pose):
        """仿真控制消费者线程：周期性获取角度数据并控制仿真
        
        Args:
            on_pose: 回调函数，接收动作向量
        """
        period = max(1.0 / self.rate, 1e-6)
        next_time = time.monotonic()

        # 安全获取动作空间维度
        try:
            action_shape = self.env.action_space.shape[0] if self.env.action_space is not None else 0
        except (AttributeError, TypeError, IndexError):
            action_shape = 0
        print("Action Space Shape:", action_shape)
        
        while not self.stop_event.is_set():
            pose = self.get_latest_arm_pos(timeout=0.0)
            if pose is not None:
                try:
                    action = self.convert_pose_to_action(pose)
                    on_pose(action)
                except Exception as e:
                    print(f"仿真控制回调异常: {e}")
                    
            # 维持固定频率
            next_time += period
            sleep_dt = next_time - time.monotonic()
            if sleep_dt > 0:
                time.sleep(sleep_dt)
            else:
                next_time = time.monotonic()
    
    def run(self):
        """启动遥操作系统"""
        print("启动角度读取线程...")
        self.produce_thread.start()
        print("启动仿真控制线程...")
        self.consume_thread.start()
        
        try: 
            print("系统运行中，按 Ctrl+C 停止...")
            while True: 
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("收到中断信号，准备停止...")
        finally:
            self.stop_event.set()
            self.produce_thread.join(timeout=2.0)
            self.consume_thread.join(timeout=2.0)
            print("已停止所有线程")
            self.env.close()
            self.ser.close()
            print("资源清理完成")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='机械臂遥操作仿真程序',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--robot', '-r', 
        type=str, 
        default='so100',
        choices=['arx-x5', 'so100', 'xarm6_robotiq', 'panda', 'x_fetch', 'piper', 'widowx250s'],
        help='选择要控制的机械臂类型'
    )
    parser.add_argument(
        '--scene', '-s', 
        type=str, 
        default='ReplicaCAD_SceneManipulation-v1',
        help='仿真场景名称'
    )
    parser.add_argument(
        '--rate', 
        type=float, 
        default=50.0,
        help='控制频率 (Hz)'
    )
    parser.add_argument(
        '--serial-port', 
        type=str, 
        default='/dev/ttyUSB0',
        help='串口设备路径'
    )
    
    args = parser.parse_args()
    
    # 显示启动信息
    print("=" * 60)
    print("    机械臂遥操作仿真系统")
    print("=" * 60)
    print(f"机械臂类型: {args.robot}")
    print(f"仿真场景:   {args.scene}")
    print(f"控制频率:   {args.rate} Hz")
    print(f"串口设备:   {args.serial_port}")
    print("-" * 60)
    
    # 创建并运行仿真实例
    try:
        sim = ServoTeleoperatorSim(scene=args.scene, robot_uids=args.robot, serial_port=args.serial_port)
        sim.rate = args.rate
        sim.run()
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()