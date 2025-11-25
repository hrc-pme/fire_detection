#!/usr/bin/env python3
import os
import subprocess
import tkinter as tk
from tkinter import messagebox
from datetime import datetime

# ====== CONFIGURATION ======
BAG_DIR = '/workspace/bags'
DATASET_NAME = 'fire_dataset'
TOPICS = [
    '/detection/fire',
    '/detection/image',
    '/stretch/cmd_vel',
    '/camera/camera/color/camera_info',
    '/camera/camera/color/image_raw',
    '/camera/camera/depth/camera_info',
    '/camera/camera/depth/image_rect_raw',
    '/tf',
    '/tf_static',
    '/relative_pose_stamped',
]
# ==========================

class RecorderGUI:
    def __init__(self, root):
        self.root = root
        self.root.title('ROS2 Bag Recorder')
        self.proc = None
        self.bag_path = self.get_bag_path()

        tk.Label(root, text=f'Bag will be saved to:').pack(pady=5)
        self.path_label = tk.Label(root, text=self.bag_path, fg='blue')
        self.path_label.pack(pady=5)

        self.start_btn = tk.Button(root, text='Start Recording', command=self.start_record, width=20, bg='green', fg='white')
        self.start_btn.pack(pady=10)
        self.stop_btn = tk.Button(root, text='Stop/Exit', command=self.stop_or_exit, width=20, bg='red', fg='white', state='normal')
        self.stop_btn.pack(pady=5)
        self.is_recording = False

    def get_bag_path(self):
        dt = datetime.now().strftime('%Y%m%d_%H%M%S')
        bag_name = f'{DATASET_NAME}_{dt}'
        return os.path.abspath(os.path.join(BAG_DIR, bag_name))

    def start_record(self):
        if self.proc is not None or self.is_recording:
            messagebox.showwarning('Warning', 'Recording already started!')
            return
        os.makedirs(BAG_DIR, exist_ok=True)
        cmd = ['ros2', 'bag', 'record', '--storage', 'mcap', '-o', self.bag_path] + TOPICS
        self.proc = subprocess.Popen(cmd)
        self.start_btn.config(state='disabled')
        self.is_recording = True
        self.path_label.config(text=self.bag_path)

    def stop_or_exit(self):
        if self.is_recording and self.proc is not None:
            self.proc.terminate()
            self.proc.wait()
            self.proc = None
            self.is_recording = False
            messagebox.showinfo('Info', f'Saved to:\n{self.bag_path}')
            self.start_btn.config(state='normal')
            # 不退出，僅停止錄製
        else:
            self.root.quit()

if __name__ == '__main__':
    root = tk.Tk()
    gui = RecorderGUI(root)
    root.mainloop()
