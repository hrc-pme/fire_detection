#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os


MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_nano_111.pt')
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.1
COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/depth/image_rect_raw'
OUTPUT_TOPIC = '/detection/fire'
AREA_UPPER_LIMIT = 50000


class FireDetectionNode(Node):

    def get_valid_zc(self, depth_img, cx, cy):
        h, w = depth_img.shape[:2]
        cx_i = int(np.clip(round(cx), 0, w-1))
        cy_i = int(np.clip(round(cy), 0, h-1))
        zc = float(depth_img[cy_i, cx_i])
        if zc > 0:
            return zc

        # 3x3 補插
        zs = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx = int(np.clip(cx_i + dx, 0, w-1))
                ny = int(np.clip(cy_i + dy, 0, h-1))
                val = float(depth_img[ny, nx])
                if val > 0:
                    zs.append(val)

        return float(np.mean(zs)) if zs else 0.0


    def __init__(self):
        super().__init__('fire_detection_node')

        self.model = YOLO(MODEL_PATH)
        self.bridge = CvBridge()
        self.latest_depth = None

        self.image_sub = self.create_subscription(Image, COLOR_TOPIC, self.image_callback, 10)
        self.depth_sub = self.create_subscription(Image, DEPTH_TOPIC, self.depth_callback, 10)

        self.fire_pub = self.create_publisher(Point, OUTPUT_TOPIC, 10)
        self.image_pub = self.create_publisher(Image, '/detection/image', 10)

        self.frame_count = 0
        self.detection_count = 0

        self.get_logger().info('Fire Detection Node initialized (NO rotation mode)')


    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, 'passthrough')
        except Exception as e:
            self.get_logger().error(f'Depth error: {e}')


    def image_callback(self, msg):
        try:
            # ---- 1) 直接取得原始影像（無旋轉） ----
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            display_image = cv_image.copy()
            self.frame_count += 1

            # ---- 2) YOLO detection ----
            results = self.model.predict(
                cv_image,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                verbose=False
            )

            depth_img = self.latest_depth
            detections = []

            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = float(box.conf[0].cpu().numpy())
                    cls = int(box.cls[0].cpu().numpy())

                    area = abs((x2 - x1) * (y2 - y1))
                    if area > AREA_UPPER_LIMIT:
                        continue

                    cx = (x1 + x2) / 2
                    cy = (y1 + y2) / 2

                    # ---- 3) depth ----
                    z_center = None
                    z_corners = []

                    if depth_img is not None:
                        h, w = depth_img.shape[:2]

                        # 角點深度
                        for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                            px_i = int(np.clip(round(px), 0, w-1))
                            py_i = int(np.clip(round(py), 0, h-1))
                            z_corners.append(float(depth_img[py_i, px_i]))

                        z_center = self.get_valid_zc(depth_img, cx, cy)

                    detections.append((x1, y1, x2, y2, cx, cy, z_center, conf))

                    # ---- Publish fire center ----
                    pt = Point()
                    pt.x = float(cx)
                    pt.y = float(cy)
                    pt.z = float(z_center) if z_center is not None else 0.0
                    self.fire_pub.publish(pt)
                    self.detection_count += 1

            # ---- 4) Draw on original frame ----
            for x1, y1, x2, y2, cx, cy, zc, conf in detections:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                # red box
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

                # green center
                cv2.circle(display_image, (int(cx), int(cy)), 5, (0, 255, 0), -1)

                # label
                label = f"Fire {conf:.2f} Z:{zc:.0f}"
                cv2.putText(display_image, label, (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # ---- 5) GUI 與 ROS output 完全一致 ----
            cv2.imshow("Fire Detection", display_image)
            cv2.waitKey(1)

            # ---- 6) Convert to RGB before publishing ----
            rgb_img = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)

            out_msg = self.bridge.cv2_to_imgmsg(rgb_img, encoding='rgb8')
            out_msg.header = msg.header
            self.image_pub.publish(out_msg)

        except Exception as e:
            self.get_logger().error(f"image_callback error: {e}")


def main(args=None):
    rclpy.init(args=args)
    node = FireDetectionNode()
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
