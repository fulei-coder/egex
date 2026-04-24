#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import pyrealsense2 as rs
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import numpy as np

class CameraNode(Node):
    def __init__(self):
        super().__init__('multi_cam_node')
        self.bridge = CvBridge()

        # 配置第一个 RealSense 相机（使用序列号）
        self.pipeline_1 = rs.pipeline()
        config_1 = rs.config()
        config_1.enable_device('338622073582')  # 第一个相机的序列号
        config_1.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline_1.start(config_1)

        # 配置第二个 RealSense 相机（使用序列号）
        self.pipeline_2 = rs.pipeline()
        config_2 = rs.config()
        config_2.enable_device('148522073685')  # 第二个相机的序列号
        config_2.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.pipeline_2.start(config_2)

        # 发布图像话题
        self.pub_1 = self.create_publisher(Image, '/cam_1', 30)
        self.pub_2 = self.create_publisher(Image, '/cam_2', 30)

        # 定时器以10Hz频率发布图像
        self.timer = self.create_timer(0.1, self.publish_images)  # 10Hz 可调

    def publish_images(self):
        # 获取第一个相机的图像数据
        frames_1 = self.pipeline_1.wait_for_frames()  # 获取帧
        color_frame_1 = frames_1.get_color_frame()  # 获取颜色帧
        if not color_frame_1:
            self.get_logger().warn("Failed to capture image from camera 1.")
        else:
            frame_1 = np.asanyarray(color_frame_1.get_data())  # 转换为 NumPy 数组
            ros_img_1 = self.bridge.cv2_to_imgmsg(frame_1, encoding='bgr8')
            self.pub_1.publish(ros_img_1)
            self.get_logger().info("Published image from camera 1")
            
            # Visualize the image from camera 1
            # self.visualize_image(frame_1, "Camera 1")

        # 获取第二个相机的图像数据
        frames_2 = self.pipeline_2.wait_for_frames()  # 获取帧
        color_frame_2 = frames_2.get_color_frame()  # 获取颜色帧
        if not color_frame_2:
            self.get_logger().warn("Failed to capture image from camera 2.")
        else:
            frame_2 = np.asanyarray(color_frame_2.get_data())  # 转换为 NumPy 数组
            ros_img_2 = self.bridge.cv2_to_imgmsg(frame_2, encoding='bgr8')
            self.pub_2.publish(ros_img_2)
            self.get_logger().info("Published image from camera 2")
            
            # Visualize the image from camera 2
            # self.visualize_image(frame_2, "Camera 2")

    def visualize_image(self, image, window_name):
        # Display the image using OpenCV
        cv2.imshow(window_name, image)
        cv2.waitKey(1)  # Wait for a key press to update the window

    def __del__(self):
        # 停止流
        self.pipeline_1.stop()
        self.pipeline_2.stop()
        self.get_logger().info("Cameras released.")
        cv2.destroyAllWindows()  # Close all OpenCV windows when done

def main(args=None):
    rclpy.init(args=args)
    node = CameraNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
