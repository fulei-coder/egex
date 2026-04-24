#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import serial
import time
import re
import numpy as np
from std_msgs.msg import Float64MultiArray

class ServoReaderNode(Node):
    def __init__(self):
        super().__init__("servo_reader_node")
        self.pub = self.create_publisher(Float64MultiArray, '/servo_angles', 10)
        self.rate = self.create_rate(50)
        self.SERIAL_PORT = '/dev/ttyUSB0'

        # 声明参数并获取值
        self.declare_parameter("baudrate", 115200)
        self.BAUDRATE = self.get_parameter("baudrate").get_parameter_value().integer_value
        self.ser = serial.Serial(self.SERIAL_PORT, self.BAUDRATE, timeout=0.1)
        self.get_logger().info("串口已打开")

        self.gripper_range = 0.48
        self.zero_angles = [0.0] * 7
        self.valid_servos = [0, 1, 2]  # 只使用舵机0-2
        self._init_servos()

    def send_command(self, cmd):
        self.ser.write(cmd.encode('ascii'))
        time.sleep(0.008)
        return self.ser.read_all().decode('ascii', errors='ignore')

    def pwm_to_angle(self, response_str, pwm_min=500, pwm_max=2500, angle_range=270):
        match = re.search(r'P(\d{4})', response_str)
        if not match:
            return None
        pwm_val = int(match.group(1))
        pwm_span = pwm_max - pwm_min
        angle = (pwm_val - pwm_min) / pwm_span * angle_range
        return angle

    def _init_servos(self):
        self.send_command('#000PVER!')
        for i in self.valid_servos:  # 只初始化有效的舵机
            self.send_command("#000PCSK!")
            self.send_command(f'#{i:03d}PULK!')
            response = self.send_command(f'#{i:03d}PRAD!')
            angle = self.pwm_to_angle(response.strip())
            self.zero_angles[i] = angle if angle is not None else 0.0
        self.get_logger().info("舵机初始角度校准完成")

    def run(self):
        angle_offset = [0.0] * 7  # 当前发布出去的角度
        target_angle_offset = [0.0] * 7  # 每个舵机的目标角度
        num_interp = 5  # 插值步数
        step_size = 1  # 最小变化量阈值

        while rclpy.ok():
            for i in self.valid_servos:  # 只读取有效的舵机
                response = self.send_command(f'#{i:03d}PRAD!')
                angle = self.pwm_to_angle(response.strip())
                if angle is not None:
                    new_angle = angle - self.zero_angles[i]
                    if abs(new_angle - target_angle_offset[i]) > step_size:
                        target_angle_offset[i] = new_angle
                else:
                    self.get_logger().warn(f"舵机 {i} 回传异常: {response.strip()}")

            # 插值逼近目标角度
            for step in range(num_interp):
                for i in range(7):
                    delta = target_angle_offset[i] - angle_offset[i]
                    angle_offset[i] += delta * 0.2  # 惰性插值，系数 < 1 可调节平滑度
                msg = Float64MultiArray()
                msg.data = angle_offset
                self.pub.publish(msg)
                self.rate.sleep()

def main(args=None):
    rclpy.init(args=args)
    node = ServoReaderNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main() 