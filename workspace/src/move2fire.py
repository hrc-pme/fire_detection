#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, Point
import math
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2

# 參數
VEL_TOPIC = '/stretch/cmd_vel'
FIRE_TOPIC = '/detection/fire'
FORWARD_SPEED = 0.1  # m/s
ANGULAR_SPEED = math.radians(30)  # rad/s
STOP_DIST = 800  # mm
CENTER_X = 240  # 假設影像寬度 640
MIN_ANGULAR_THRESHOLD = 5  # pixel

class Move2FireNode(Node):
    def __init__(self):
        super().__init__('move2fire_node')
        self.publisher = self.create_publisher(Twist, VEL_TOPIC, 10)
        self.subscription = self.create_subscription(Point, FIRE_TOPIC, self.fire_callback, 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.latest_fire = None
        self.image_pub = self.create_publisher(Image, '/detection/image', 10)
        self.bridge = CvBridge()
        self.latest_image = None
        self.image_sub = self.create_subscription(Image, '/camera/camera/color/image_raw', self.image_callback, 10)
        self.bbox = None
        self.get_logger().info(f'Subscribed to: {FIRE_TOPIC}, publishing to: {VEL_TOPIC} and /detection/image')

    def image_callback(self, msg):
        # Save latest image for publishing
        self.latest_image = msg

    def fire_callback(self, msg):
        # 只記錄最近一次 fire
        if msg.z > 0:
            if self.latest_image is not None:
                cv_image = self.bridge.imgmsg_to_cv2(self.latest_image, desired_encoding='bgr8')
                # 只畫中心點
                cv2.circle(cv_image, (int(msg.x), int(msg.y)), 20, (0,0,255), 3)
                out_msg = self.bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
                out_msg.header = self.latest_image.header
                self.image_pub.publish(out_msg)
            self.latest_fire = msg

    def timer_callback(self):
        twist = Twist()
        reason = None
        if self.latest_fire is not None:
            x = self.latest_fire.x
            z = self.latest_fire.z
            offset = x - CENTER_X
            if z > 0:
                # 旋轉速度指數縮放，完全對齊中心點才停止旋轉
                if offset != 0:
                    exp_scale = min(math.exp(abs(offset) / CENTER_X) - 1, 1.0)
                    twist.angular.z = (-ANGULAR_SPEED if offset > 0 else ANGULAR_SPEED) * exp_scale
                else:
                    twist.angular.z = 0.0
                if z > STOP_DIST:
                    twist.linear.x = FORWARD_SPEED
                    self.get_logger().info(f'Approaching fire: x={x:.1f}, z={z:.1f}, offset={offset:.1f}, v={twist.linear.x}, w={twist.angular.z}')
                else:
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                    reason = f'Fire within {STOP_DIST:.0f}mm, stop. x={x:.1f}, z={z:.1f}'
            else:
                twist.linear.x = 0.0
                twist.angular.z = 0.0
                reason = f'Fire z invalid (z={z:.1f}), stop.'
        else:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            reason = 'No fire detected, stop.'
        self.publisher.publish(twist)
        if reason:
            self.get_logger().info(f'[STOP] {reason}')

def main(args=None):
    rclpy.init(args=args)
    try:
        node = Move2FireNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        rclpy.shutdown()

if __name__ == '__main__':
    main()
