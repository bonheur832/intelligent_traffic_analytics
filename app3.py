import tkinter as tk
from tkinter import filedialog
import torch
import cv2
from PIL import Image, ImageTk
from collections import defaultdict
import csv

model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')
class_counts = defaultdict(int)

def select_video():
    video_path = filedialog.askopenfilename()
    if video_path:
        process_video(video_path)

def process_video(path):
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        df = results.pandas().xyxy[0]
        for name in df['name']:
            class_counts[name] += 1
        # Display frame
        annotated = results.render()[0]
        img = Image.fromarray(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        img_tk = ImageTk.PhotoImage(image=img)
        video_label.config(image=img_tk)
        video_label.image = img_tk
        root.update()
    cap.release()

def export_summary():
    file_path = filedialog.asksaveasfilename(defaultextension=".csv")
    if file_path:
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Count'])
            for cls, count in class_counts.items():
                writer.writerow([cls, count])

root = tk.Tk()
tk.Button(root, text="Upload Video", command=select_video).pack()
tk.Button(root, text="Export Summary", command=export_summary).pack()
video_label = tk.Label(root)
video_label.pack()
root.mainloop()
