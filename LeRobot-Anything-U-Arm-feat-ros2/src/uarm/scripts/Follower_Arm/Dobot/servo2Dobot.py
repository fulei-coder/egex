#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from api import Bestman_Real_CR5
import time
import math
import numpy as np

class RobotControlNode(Node):
    def __init__(self):
        super().__init__('dobot_teleop_node')
        self.robot_ip = "192.168.5.1"  # 请替换为实际的机器人IP地址
        self.robot = Bestman_Real_CR5(ip=self.robot_ip, text_log=True)

        # 发布当前机器人状态的topic
        self.state_pub = self.create_publisher(Float64MultiArray, '/robot_action', 10)

        # 订阅控制指令的topic
        self.subscription = self.create_subscription(
            Float64MultiArray,
            '/servo_angles',
            self.joint_cmd_callback,
            10)

        # 定时器来发布机器人状态
        self.rate = self.create_rate(10)  # 10hz

        self.init_qpos = np.array([180,-20,-96,62,93,205])
        self.robot.move_arm_to_joint_angles(self.init_qpos,wait=True)

    def joint_cmd_callback(self, msg):
        """处理接收到的关节控制命令"""
        try:
            # 打印接收到的数据
            print(f"msg.data:{msg.data[:6]}")
            servo_angles = np.array(msg.data)

            servo_angles[1] = -servo_angles[1] #adjustment

            joint_angles = (servo_angles[:6]+self.init_qpos)

            
            # joint_angles = (np.array(msg.data[:6]) + np.array(self.init_qpos)).tolist()
            
            self.robot.move_arm_to_joint_angles_servo(joint_angles,t=0.1,gain=250)
            # self.get_logger().info(f"已移动机器人到指定关节角度: {joint_angles}")
        except Exception as e:
            self.get_logger().error(f"移动机器人失败: {e}")

    def run(self):
        """主循环，发布状态并持续监听控制指令"""
        while rclpy.ok():
            self.rate.sleep()

def main(args=None):
    rclpy.init(args=args)
    robot_node = RobotControlNode()
    try:
        robot_node.run()
    except KeyboardInterrupt:
        pass
    finally:
        robot_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
