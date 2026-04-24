import gymnasium as gym
import mani_skill.envs  # 必须导入以注册所有 env/agent
import numpy as np
import argparse
import time
import sapien
from mani_skill.utils import sapien_utils

class StaticRobotViewer:
    """静态机械臂查看器
    
    在仿真环境中显示指定类型的机械臂，支持多种机械臂类型：
    arx-x5, so100, xarm6_robotiq, panda, x_fetch, unitree_h1
    """
    
    def __init__(self, scene: str, robot_uids: str, pose_name: str = "default"):
        """初始化静态机械臂查看器
        
        Args:
            scene: 仿真场景名称
            robot_uids: 机械臂类型标识符
            pose_name: 姿态名称（default, standing, t_pose等）
        """
        self.scene = scene
        self.robot_uids = robot_uids
        self.pose_name = pose_name
        
        # 根据机器人类型选择控制模式
        if robot_uids == "x_fetch": 
            self.control_mode = "pd_joint_pos_dual_arm"
        elif robot_uids == "unitree_h1":
            self.control_mode = "pd_joint_delta_pos"
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
        )
        
        # 获取动作空间信息
        obs, _ = self.env.reset(seed=0)
        print(f"动作空间: {self.env.action_space}")
        print(f"观察空间: {self.env.observation_space}")
        
        # 设置机械臂姿态
        self._setup_robot_pose()
        self._setup_camera_pose()

    def _setup_camera_pose(self):
        agent = getattr(self.env.unwrapped, "agent", None)
        pose = sapien.Pose()
        if agent is not None:
            pose = agent.robot.get_pose()  # 返回 sapien.Pose
            print(f"机器人初始位置: {pose}")
        camera_pose = sapien_utils.look_at(
            [-1.4, -1.1, 1.7], pose.p
        )
        camera_viewer = getattr(self.env.unwrapped, "viewer", None)
        if camera_viewer is not None:
            print(camera_pose)
            camera_pose_arr = camera_pose.raw_pose.squeeze().cpu().numpy()
            camera_position = camera_pose_arr[:3]
            camera_quaternion = camera_pose_arr[3:]
            camera_viewer.set_camera_pose(sapien.Pose(camera_position, camera_quaternion))

    def _setup_robot_pose(self):
        """根据机器人类型和姿态名称设置机械臂姿态"""
        try:
            agent = getattr(self.env.unwrapped, "agent", None)
            if agent is None:
                print("警告：无法获取机器人代理")
                return
                
            if self.robot_uids == "unitree_h1":
                # H1人形机器人特殊处理
                if self.pose_name == "standing":
                    if hasattr(agent, "keyframes") and "standing" in agent.keyframes:
                        standing_keyframe = agent.keyframes["standing"]
                        agent.reset(standing_keyframe.qpos)
                        agent.robot.set_root_pose(standing_keyframe.pose)
                        print("H1 已设置为站立姿态")
                    else:
                        print("警告：H1 站立姿态不可用，使用默认姿态")
                elif self.pose_name == "t_pose":
                    if hasattr(agent, "keyframes") and "t_pose" in agent.keyframes:
                        t_pose_keyframe = agent.keyframes["t_pose"]
                        agent.reset(t_pose_keyframe.qpos)
                        agent.robot.set_root_pose(t_pose_keyframe.pose)
                        print("H1 已设置为T型姿态")
                    else:
                        print("警告：H1 T型姿态不可用，使用默认姿态")
                else:
                    print("H1 使用默认姿态")
                    
            elif self.robot_uids == "x_fetch":
                # Fetch双臂机器人
                if self.pose_name == "home":
                    # 设置双臂到home位置
                    home_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    agent.reset(home_pose)
                    print("Fetch 已设置为home姿态")
                else:
                    print("Fetch 使用默认姿态")
                    
            elif self.robot_uids == "panda":
                # Panda机械臂
                if self.pose_name == "home":
                    # 设置到home位置
                    home_pose = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.7854, 0.0])
                    agent.reset(home_pose)
                    print("Panda 已设置为home姿态")
                else:
                    print("Panda 使用默认姿态")
                    
            elif self.robot_uids == "xarm6_robotiq":
                # XArm6机械臂
                if self.pose_name == "home":
                    # 设置到home位置
                    home_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    agent.reset(home_pose)
                    print("XArm6 已设置为home姿态")
                else:
                    print("XArm6 使用默认姿态")
                    
            elif self.robot_uids == "arx-x5":
                # ARX-X5机械臂
                if self.pose_name == "home":
                    # 设置到home位置
                    home_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    agent.reset(home_pose)
                    print("ARX-X5 已设置为home姿态")
                else:
                    print("ARX-X5 使用默认姿态")
                    
            elif self.robot_uids == "so100":
                # SO100机械臂
                if self.pose_name == "home":
                    # 设置到home位置
                    home_pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
                    agent.reset(home_pose)
                    print("SO100 已设置为home姿态")
                else:
                    print("SO100 使用默认姿态")
                    
        except Exception as e:
            print(f"设置机器人姿态时出错: {e}")
    
    def get_robot_info(self):
        """获取机器人信息"""
        try:
            agent = getattr(self.env.unwrapped, "agent", None)
            if agent is None:
                return "无法获取机器人信息"
                
            info = f"机器人类型: {self.robot_uids}\n"
            info += f"控制模式: {self.control_mode}\n"
            
            if hasattr(agent, "robot"):
                robot = agent.robot
                info += f"机器人名称: {robot.name}\n"
                
                # 获取关节信息
                if hasattr(robot, "get_active_joints"):
                    joints = robot.get_active_joints()
                    info += f"关节数量: {len(joints)}\n"
                    for i, joint in enumerate(joints):
                        info += f"  关节{i}: {joint.name}\n"
                        
                # 获取当前关节位置
                if hasattr(agent, "get_qpos"):
                    qpos = agent.get_qpos()
                    info += f"当前关节位置: {qpos}\n"
                    
            return info
            
        except Exception as e:
            return f"获取机器人信息时出错: {e}"
    
    def run(self, duration: float = None):
        """运行静态机械臂查看器
        
        Args:
            duration: 运行时长（秒），None表示无限运行
        """
        print("=" * 60)
        print("    静态机械臂查看器")
        print("=" * 60)
        print(f"机械臂类型: {self.robot_uids}")
        print(f"仿真场景:   {self.scene}")
        print(f"姿态名称:   {self.pose_name}")
        print(f"控制模式:   {self.control_mode}")
        print("-" * 60)
        
        # 显示机器人信息
        print(self.get_robot_info())
        print("-" * 60)
        
        start_time = time.time()
        
        try:
            print("开始渲染，按 Ctrl+C 停止...")
            
            while True:
                # 渲染当前帧
                self.env.render()
                
                # 检查是否超时
                if duration is not None and (time.time() - start_time) > duration:
                    print(f"运行时间达到 {duration} 秒，自动停止")
                    break
                    
                # 短暂延迟以控制渲染频率
                time.sleep(0.033)  # 约30 FPS
                
        except KeyboardInterrupt:
            print("收到中断信号，准备停止...")
        finally:
            self.env.close()
            print("资源清理完成")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='静态机械臂查看器',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--robot', '-r', 
        type=str, 
        default='so100',
        choices=['arx-x5', 'so100', 'xarm6_robotiq', 'panda', 'x_fetch', 'unitree_h1'],
        help='选择要显示的机械臂类型'
    )
    parser.add_argument(
        '--scene', '-s', 
        type=str, 
        default='ReplicaCAD_SceneManipulation-v1',
        help='仿真场景名称'
    )
    parser.add_argument(
        '--pose', '-p',
        type=str,
        default='default',
        choices=['default', 'home', 'standing', 't_pose'],
        help='机械臂姿态名称'
    )
    parser.add_argument(
        '--duration', '-d',
        type=float,
        default=None,
        help='运行时长（秒），默认无限运行'
    )
    
    args = parser.parse_args()
    
    # 创建并运行查看器
    try:
        viewer = StaticRobotViewer(
            scene=args.scene, 
            robot_uids=args.robot, 
            pose_name=args.pose
        )
        viewer.run(duration=args.duration)
    except Exception as e:
        print(f"程序运行出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 