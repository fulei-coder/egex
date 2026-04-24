#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
from std_msgs.msg import Float64MultiArray
from xarm.wrapper import XArmAPI

class ArmInterfaceNode(Node):
    def __init__(self):
        super().__init__('arm_inter')
        self.get_logger().info("Connecting to xArm...")

        # initialize xArm
        self.declare_parameter("robot_ip", "192.168.1.199")
        robot_ip = self.get_parameter("robot_ip").get_parameter_value().string_value  #TODO: change IP to yours
        self.arm = XArmAPI(robot_ip)
        self.arm.motion_enable(True)
        self.arm.set_mode(6)  
        self.arm.set_state(0)  

        self.get_logger().info("xArm connected.")

        # Initialize ROS publishers
        self.state_pub = self.create_publisher(Float64MultiArray, '/robot_state', 10)
        self.timer = self.create_timer(0.1, self.publish_state)

    def publish_state(self):
        joint_angles = self.arm.get_servo_angle()
        gripper_pos = self.arm.get_gripper_position()

        if joint_angles and joint_angles[0] == 0:
            angles = joint_angles[1][:6]
        else:
            angles = [0.0] * 6

        if gripper_pos and gripper_pos[0] == 0:
            gpos = gripper_pos[1] 
        else:
            gpos = 0.0

        full_state = angles + [gpos]

        msg = Float64MultiArray()
        msg.data = full_state
        self.state_pub.publish(msg)
        self.get_logger().info(f"Published robot state: {msg.data}")

def main(args=None):
    rclpy.init(args=args)
    node = ArmInterfaceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()