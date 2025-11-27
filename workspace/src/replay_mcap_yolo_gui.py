#!/usr/bin/env python3
import rosbag2_py
import cv2
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message
from ultralytics import YOLO
import os

# ====== 自行修改 ======
BAG_PATH = "/workspace/bags/fire_dataset_20251126_005258/fire_dataset_20251126_005258_0.mcap"
IMAGE_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"
MODEL_PATH = "/workspace/models/best_nano_111.pt"

CONF_TH = 0.35
IOU_TH = 0.1
AREA_LIMIT = 60000          # 過大 bbox 不顯示
ROTATE = True               # 旋轉 90°（順時鐘）
OUTPUT_VIDEO = "output.mp4" # 錄影輸出檔名

# ===============================================================

def rotate_img_90(img):
    return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)

def get_valid_depth(depth_img, cx, cy):
    """9點平均補插深度"""
    h, w = depth_img.shape[:2]
    cx_i = int(np.clip(round(cx), 0, w-1))
    cy_i = int(np.clip(round(cy), 0, h-1))

    z = float(depth_img[cy_i, cx_i])
    if z > 0:
        return z

    zs = []
    for dx in [-1,0,1]:
        for dy in [-1,0,1]:
            nx = int(np.clip(cx_i + dx, 0, w-1))
            ny = int(np.clip(cy_i + dy, 0, h-1))
            v = float(depth_img[ny, nx])
            if v > 0:
                zs.append(v)

    return float(np.mean(zs)) if zs else 0.0


def replay_mcap_with_yolo():
    model = YOLO(MODEL_PATH)
    bridge = CvBridge()

    # rosbag2 reader
    converter = rosbag2_py.ConverterOptions("", "")
    reader = rosbag2_py.SequentialReader()
    storage = rosbag2_py.StorageOptions(uri=BAG_PATH, storage_id="mcap")
    reader.open(storage, converter)

    topics = reader.get_all_topics_and_types()
    type_map = {t.name: t.type for t in topics}

    if IMAGE_TOPIC not in type_map:
        print(f"[ERROR] image topic not found: {IMAGE_TOPIC}")
        return

    has_depth = DEPTH_TOPIC in type_map
    if not has_depth:
        print("[WARN] No depth topic found → depth=0 always")

    img_type = get_message(type_map[IMAGE_TOPIC])
    if has_depth:
        depth_type = get_message(type_map[DEPTH_TOPIC])

    video_writer = None

    print("[INFO] Start replay. Press q or ESC to exit.")

    latest_depth = None

    while reader.has_next():
        topic, data, t = reader.read_next()

        # ---- 深度影像 ----
        if topic == DEPTH_TOPIC:
            depth_msg = deserialize_message(data, depth_type)
            latest_depth = bridge.imgmsg_to_cv2(depth_msg, 'passthrough')
            if ROTATE:
                latest_depth = rotate_img_90(latest_depth)
            continue

        # ---- 彩色影像 ----
        if topic != IMAGE_TOPIC:
            continue

        img_msg = deserialize_message(data, img_type)
        cv_img = bridge.imgmsg_to_cv2(img_msg, 'bgr8')

        if ROTATE:
            cv_img = rotate_img_90(cv_img)

        # ---- YOLO 偵測 ----
        results = model.predict(cv_img, conf=CONF_TH, iou=IOU_TH, verbose=False)

        display = cv_img.copy()

        # ---- 處理每個 detection ----
        for r in results:
            if r.boxes is None:
                continue

            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = float(box.conf[0].cpu().numpy())

                area = abs((x2 - x1) * (y2 - y1))
                if area > AREA_LIMIT:
                    continue

                # 中心點
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                # 深度
                depth_value = 0.0
                if latest_depth is not None:
                    depth_value = get_valid_depth(latest_depth, cx, cy)

                # ---- 繪圖 ----
                cv2.rectangle(display, (int(x1), int(y1)), (int(x2), int(y2)),
                              (0,0,255), 2)

                cv2.circle(display, (int(cx), int(cy)), 6, (0,255,0), -1)

                label = f"Z:{depth_value:.0f}"
                cv2.putText(display, label, (int(x1), int(y1)-8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # ---- 初始化 VideoWriter ----
        if video_writer is None:
            h, w = display.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, 30.0, (w, h))

        video_writer.write(display)

        # ---- 顯示 GUI ----
        cv2.imshow("MCAP YOLO Replay", display)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()
    print(f"[INFO] Replay done. Video saved → {OUTPUT_VIDEO}")


if __name__ == "__main__":
    replay_mcap_with_yolo()
