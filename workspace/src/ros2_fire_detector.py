#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import Header
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import torch
import os


# ==== 全域參數設定 ====
MODEL_PATH = os.path.join(os.path.dirname(__file__), '../models/best_nano_111.pt')
CONFIDENCE_THRESHOLD = 0.35
IOU_THRESHOLD = 0.1
COLOR_TOPIC = '/camera/camera/color/image_raw'
DEPTH_TOPIC = '/camera/camera/depth/image_rect_raw'
OUTPUT_TOPIC = '/detection/fire'
AREA_UPPER_LIMIT = 50000  # 最大 bbox 面積

class FireDetectionNode(Node):
    def get_valid_zc(self, depth_img, cx, cy):
        """
        取得中心點 Zc，若為 0，則用 3x3 區域非零平均值，否則回傳 0。
        """
        h, w = depth_img.shape[:2]
        cx_i = int(np.clip(round(cx), 0, w-1))
        cy_i = int(np.clip(round(cy), 0, h-1))
        zc = float(depth_img[cy_i, cx_i])
        if zc > 0:
            return zc
        # 取 3x3 區域非零平均
        zs = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                nx = int(np.clip(cx_i + dx, 0, w-1))
                ny = int(np.clip(cy_i + dy, 0, h-1))
                val = float(depth_img[ny, nx])
                if val > 0:
                    zs.append(val)
        if zs:
            return float(np.mean(zs))
        return 0.0

    def __init__(self):
            """
            取得中心點 Zc，若為 0，則用 3x3 區域非零平均值，否則回傳 0。
            """
            h, w = depth_img.shape[:2]
            cx_i = int(np.clip(round(cx), 0, w-1))
            cy_i = int(np.clip(round(cy), 0, h-1))
            zc = float(depth_img[cy_i, cx_i])
            if zc > 0:
                return zc
            # 取 3x3 區域非零平均
            zs = []
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    nx = int(np.clip(cx_i + dx, 0, w-1))
                    ny = int(np.clip(cy_i + dy, 0, h-1))
                    val = float(depth_img[ny, nx])
                    if val > 0:
                        zs.append(val)
            if zs:
                return float(np.mean(zs))
            return 0.0
    def __init__(self):
        super().__init__('fire_detection_node')
        # 參數
        self.model_path = MODEL_PATH
        self.conf_threshold = CONFIDENCE_THRESHOLD
        self.iou_threshold = IOU_THRESHOLD
        self.input_topic = COLOR_TOPIC
        self.depth_topic = DEPTH_TOPIC
        self.output_topic = OUTPUT_TOPIC

        # YOLO
        self.get_logger().info(f'Loading YOLO model from {self.model_path}...')
        self.model = YOLO(self.model_path)
        self.get_logger().info(f'Model loaded successfully on {torch.device("cpu")}')

        # CV Bridge
        self.bridge = CvBridge()

        # 最新的 depth image
        self.latest_depth = None

        # 訂閱彩色影像
        self.image_sub = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            10
        )
        # 訂閱深度影像
        self.depth_sub = self.create_subscription(
            Image,
            self.depth_topic,
            self.depth_callback,
            10
        )
        # fire detection publisher
        self.fire_pub = self.create_publisher(Point, self.output_topic, 10)

        self.frame_count = 0
        self.detection_count = 0

        self.get_logger().info('Fire Detection Node initialized')
        self.get_logger().info(f'Subscribed to: {self.input_topic} (color) & {self.depth_topic} (depth)')
        self.get_logger().info(f'Publishing to: {self.output_topic}')
        self.get_logger().info(f'Confidence threshold: {self.conf_threshold}')
        self.get_logger().info(f'IOU threshold: {self.iou_threshold}')

    def depth_callback(self, msg):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        except Exception as e:
            self.get_logger().error(f'Error processing depth image: {str(e)}')
    
    def image_callback(self, msg):
        try:
            # Convert ROS Image to OpenCV format
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            # Rotate image 90 degrees clockwise
            cv_image = cv2.rotate(cv_image, cv2.ROTATE_90_CLOCKWISE)
            self.frame_count += 1

            # Run YOLO detection
            results = self.model.predict(
                cv_image,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            # Prepare depth image
            depth_img = self.latest_depth
            if depth_img is not None:
                # 同步旋轉
                depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)

            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is None:
                    continue
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    # 計算 bbox 面積
                    area = abs((x2 - x1) * (y2 - y1))
                    if area > AREA_UPPER_LIMIT:
                        self.get_logger().info(f'Ignore bbox area {area:.1f} > {AREA_UPPER_LIMIT}')
                        continue

                    # 取得角點與中心點的深度（中心點用補插）
                    z_corners = []
                    z_center = None
                    if depth_img is not None:
                        h, w = depth_img.shape[:2]
                        # 角點深度
                        for px, py in [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]:
                            px_i = int(np.clip(round(px), 0, w-1))
                            py_i = int(np.clip(round(py), 0, h-1))
                            z = float(depth_img[py_i, px_i])
                            z_corners.append(z)
                        # 中心點深度（補插）
                        z_center = self.get_valid_zc(depth_img, center_x, center_y)

                    det = {
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'center': (center_x, center_y),
                        'confidence': float(conf),
                        'class': cls,
                        'z_corners': z_corners,
                        'z_center': z_center
                    }
                    detections.append(det)

                    # Publish detection point (center)
                    point_msg = Point()
                    point_msg.x = float(center_x)
                    point_msg.y = float(center_y)
                    point_msg.z = float(z_center) if z_center is not None else float('nan')
                    self.fire_pub.publish(point_msg)
                    self.detection_count += 1

            # Draw bounding boxes on image
            display_image = cv_image.copy()
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                center_x, center_y = det['center']
                conf = det['confidence']
                z_corners = det['z_corners']
                z_center = det['z_center']

                # Draw bounding box
                cv2.rectangle(display_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                # Draw center point
                cv2.circle(display_image, (int(center_x), int(center_y)), 5, (0, 255, 0), -1)
                # Draw label
                label = f'Fire: {conf:.2f} Zc:{z_center if z_center is not None else "-"}'
                (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
                cv2.rectangle(display_image, (x1, y1 - label_h - 10), (x1 + label_w, y1), (0, 0, 255), -1)
                cv2.putText(display_image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Print detection info
            if detections:
                self.get_logger().info(
                    f'Frame {self.frame_count}: Detected {len(detections)} fire(s) | '
                    f'Total detections: {self.detection_count}'
                )
                for i, det in enumerate(detections):
                    bbox = det['bbox']
                    center = det['center']
                    z_corners = det['z_corners']
                    z_center = det['z_center']
                    self.get_logger().info(
                        f'  Fire #{i+1}: BBox={bbox}, Center=({center[0]:.1f},{center[1]:.1f},{z_center if z_center is not None else "-"}), '
                        f'Z_corners={z_corners}'
                    )

            # Display image with bounding boxes
            cv2.imshow('Fire Detection', display_image)
            cv2.waitKey(1)

        except Exception as e:
            self.get_logger().error(f'Error processing image: {str(e)}')


def main(args=None):
    rclpy.init(args=args)
    
    try:
        node = FireDetectionNode()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
