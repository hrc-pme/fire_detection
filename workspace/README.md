<div align="center">

# Flare Guard: Real-Time Smoke & Fire Detection with YOLO11

<img src="data/fire14.png" alt="Flare Guard Cover" width="700"/>

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
[![Telegram](https://img.shields.io/badge/Telegram-2CA5E0?logo=telegram&logoColor=white)](https://t.me/FlareGuard_bot)
[![Ultralytics](https://img.shields.io/badge/Ultralytics-181717?logo=github&logoColor=white)](https://github.com/ultralytics/ultralytics)
[![YOLOv11](https://img.shields.io/badge/YOLOv11-181717?logo=github&logoColor=white)](https://github.com/ultralytics/ultralytics)
<a href="https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11">
<img src="https://app.roboflow.com/images/download-dataset-badge.svg"></img>
</a>
<a href="https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11/model/">
<img src="https://app.roboflow.com/images/try-model-badge.svg"></img>
</a>
<a href="https://www.kaggle.com/code/sayedgamal99/smoke-fire-detection-yolon11">
<img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open in Kaggle Notebook">
</a>

</div>

## 📌 Table of Contents

1. [Introduction](#safeguard-lives-with-smart-fire-and-smoke-detection--a-new-era-of-protection)
2. [Video Demonstration](#video-demonstration)
3. [Dataset](#dataset)
4. [Training Summary](#training-summary)
5. [Example Detections](#example-detections)
6. [Installation and Usage](#installation-and-usage-)
   - [Installation](#installation)
   - [Inference](#inference)

## Safeguard Lives with Smart Fire and Smoke Detection – A New Era of Protection 🛡️

### Problem Description

Fire incidents can cause catastrophic damage and loss of life when not detected early enough. Current detection methods, which rely primarily on traditional sensors, often have delayed response times that can lead to severe consequences. This creates a critical need for faster, more reliable detection systems that can identify fire hazards at their earliest stages. To address this challenge, this project implements advanced deep learning technology to detect fires and smoke as they emerge, enabling rapid response and intervention.

### Solution

<div align="center">
<img src="data/logo.png" alt="Flare Guard Logo" width="175"/>
</div>

**Flare Guard** represents a state-of-the-art approach to fire safety monitoring. At its core, the system uses **YOLOv11**, a powerful object detection model, to continuously analyze video feeds for signs of fire and smoke in real-time. When potential threats are detected, the system immediately sends alerts through **Telegram/WhatsApp** messaging platforms, ensuring that stakeholders are notified without delay.

### Why Choose Flare Guard?

Key capabilities:
- Real-time video processing with 30-60 frames per second
- High-accuracy threat identification (80.6% Precision)
- Robust performance across various environmental conditions
- Immediate alert system with visual confirmation via Telegram and WhatsApp.
- Adaptable for multiple use cases including surveillance systems, industrial monitoring, home security, and robotic applications

The combination of speed, accuracy, and reliable notification makes Flare Guard particularly effective for environments where early fire detection is critical for safety and asset protection.


## Video Demonstration

<div align="center">
  <video width="700" src="https://github.com/user-attachments/assets/d4059264-5953-45d2-8464-b5c61e9a0caf"/>
</div>

<div align="center">
     
   ### 🔥 Fire Detection System in Action  
  This video demonstrates the system detecting fire immediately upon occurrence.  
  Alerts are sent instantly to **Telegram** and **WhatsApp**, including an attached image.  
  Thanks to **multithreading**, the system continues running without interruption.
  
</div>

## Dataset

The dataset consists of **10,463 annotated images**, available on [Roboflow](https://universe.roboflow.com/sayed-gamall/fire-smoke-detection-yolov11). This dataset is designed for training and evaluating **object detection models** tailored for real-time **fire and smoke detection**. It is suitable for:
* **Surveillance systems** (CCTV monitoring, smart security cameras)
* **Industrial safety applications** (factories, warehouses, refineries)
* **Residential safety solutions** (smart home fire detection)
* **Autonomous monitoring systems** (drones, robotics, IoT devices)

| Split      | Images | Annotations |
| ---------- | ------ | ----------- |
| Training   | 9,156  | 27,468      |
| Validation | 872    | 2,616       |
| Test       | 435    | 1,305       |

**Classes:** `Fire`, `Smoke`  
**Annotation Format:** YOLOv11-compatible bounding boxes

```python
# Download dataset via Roboflow
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("sayed-gamall").project("fire-smoke-detection-yolov11")
dataset = project.version(2).download("yolov11")
```

## Training Summary

The model was trained using **YOLOv11** on a dataset of fire and smoke images. Training stopped early due to no improvement over 20 epochs, with the best results observed at **Epoch 92**.

### Training Graph

<div align="center">
<img src="data/training_graphs.png" alt="Training Graph" width="600"/>
</div>

### Final Validation Results

| Metric        | Value     |
| ------------- | --------- |
| Precision (P) | **0.806** |
| Recall (R)    | **0.717** |
| mAP@50        | **0.770** |
| mAP@50-95     | **0.492** |

### Class-Specific Performance

| Class     | Precision | Recall | mAP@50 | mAP@50-95 |
| --------- | --------- | ------ | ------ | --------- |
| **Fire**  | 0.813     | 0.806  | 0.828  | 0.513     |
| **Smoke** | 0.800     | 0.629  | 0.711  | 0.472     |

## Example Detections

Here are examples from the test set:

<div align="center">
<img src="data/ex1.png" alt="Example 1" width="250"/>
<img src="data/ex2.png" alt="Example 2" width="250"/>
<img src="data/ex3.png" alt="Example 3" width="250"/>
</div>

## Installation and Usage 🚀

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/sayedgamal99/Real-Time-Smoke-Fire-Detection-YOLO11
   cd Real-Time-Smoke-Fire-Detection-YOLO11
   ```

2. Install the required packages:
   ```bash
   pip install ultralytics
   ```

### Inference

To perform inference with the trained model on test images, run:

```bash
yolo detect predict model=models/best_nano_111.pt source=data/house.png conf=0.35 iou=0.1
```

and there's the output:

<div align='center'>
<img src="data/ex4.png" alt="Example 1" width="700"/> 
</div>

To perform inference in real-time using a webcam:

```bash
yolo detect predict model=models/best_nano_111.pt source=0 conf=0.35 iou=0.1 show=True
```

---

<div align="center">

**Protect What Matters Most - Early Detection Saves Lives**

</div>
