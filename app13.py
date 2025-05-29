import tkinter as tk
from tkinter import filedialog, ttk, messagebox
import cv2
from PIL import Image, ImageTk
from collections import defaultdict
import csv
import time
import threading
from ultralytics import YOLO

class TrafficAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Traffic Analyzer")
        self.root.geometry("960x760")
        self.root.configure(bg="#f0f2f5")

        self.model = YOLO("yolov5s.pt")
        self.class_counts = defaultdict(int)
        self.detected_boxes = defaultdict(list)
        self.video_path = ""
        self.cap = None
        self.paused = False
        self.processing = False
        self.cancelled = False

        self.create_widgets()

    def create_widgets(self):
        title = ttk.Label(self.root, text="YOLOv5 Video Analyzer", font=("Helvetica", 18, "bold"))
        title.pack(pady=10)

        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(pady=10)

        ttk.Button(btn_frame, text="Upload Video", command=self.select_video).grid(row=0, column=0, padx=10)

        self.pause_button = ttk.Button(btn_frame, text="Pause", command=self.toggle_pause, state="disabled")
        self.pause_button.grid(row=0, column=1, padx=10)

        self.stop_button = ttk.Button(btn_frame, text="Stop", command=self.stop_processing, state="disabled")
        self.stop_button.grid(row=0, column=2, padx=10)

        ttk.Button(btn_frame, text="Export Summary", command=self.export_summary).grid(row=0, column=3, padx=10)

        self.status_label = ttk.Label(self.root, text="No video loaded", foreground="gray")
        self.status_label.pack(pady=5)

        self.progressbar = ttk.Progressbar(self.root, orient="horizontal", mode="determinate", length=500)
        self.progressbar.pack(pady=5)

        self.canvas = tk.Canvas(self.root, width=800, height=500, bg="black")
        self.canvas.pack()

    def toggle_pause(self):
        if not self.processing:
            return
        self.paused = not self.paused
        self.pause_button.config(text="Resume" if self.paused else "Pause")

    def stop_processing(self):
        self.cancelled = True
        self.status_label.config(text="❌ Processing cancelled", foreground="orange")
        self.pause_button.config(state="disabled")
        self.stop_button.config(state="disabled")

    def select_video(self):
        self.video_path = filedialog.askopenfilename(
            title="Select a Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if self.video_path:
            self.status_label.config(text=f"Loading {self.video_path.split('/')[-1]}...", foreground="blue")
            self.class_counts.clear()
            self.detected_boxes.clear()
            self.processing = True
            self.paused = False
            self.cancelled = False
            self.pause_button.config(state="normal")
            self.stop_button.config(state="normal")
            threading.Thread(target=self.process_video_thread, daemon=True).start()

    def process_video_thread(self):
        self.cap = cv2.VideoCapture(self.video_path)
        max_width, max_height = 700, 400

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / fps if fps > 0 else 0

        self.progressbar["maximum"] = total_frames
        frame_number = 0
        start_time = time.time()

        while self.cap.isOpened():
            if self.cancelled:
                break
            if self.paused:
                self.root.update()
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            frame_number += 1
            self.progressbar["value"] = frame_number
            self.status_label.config(
                text=f"Processing frame {frame_number}/{total_frames} ({(frame_number/total_frames)*100:.1f}%)"
            )
            self.root.update_idletasks()

            results = self.model(frame)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                class_name = self.model.names[cls_id]
                coords = box.xyxy[0].tolist()

                if self.is_new_detection(class_name, coords):
                    self.class_counts[class_name] += 1
                    self.detected_boxes[class_name].append(coords)

            annotated_frame = results.plot()
            frame_height, frame_width = annotated_frame.shape[:2]

            scale = min(max_width / frame_width, max_height / frame_height)
            new_width = int(frame_width * scale)
            new_height = int(frame_height * scale)
            resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

            img = Image.fromarray(cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)

            self.canvas.config(width=max_width, height=max_height)
            self.canvas.delete("all")
            x_offset = (max_width - new_width) // 2
            y_offset = (max_height - new_height) // 2
            self.canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=img_tk)
            self.canvas.image = img_tk

        self.cap.release()
        self.processing = False
        self.pause_button.config(state="disabled")
        self.stop_button.config(state="disabled")

        if self.cancelled:
            self.status_label.config(text="❌ Processing cancelled by user", foreground="orange")
        else:
            elapsed = time.time() - start_time
            self.status_label.config(
                text=f"✅ Done: {frame_number} frames in {elapsed:.2f}s (video: {duration_sec:.2f}s)",
                foreground="green"
            )
            self.show_summary()

    def is_new_detection(self, class_name, new_box, iou_threshold=0.5):
        for old_box in self.detected_boxes[class_name]:
            if self.calculate_iou(old_box, new_box) > iou_threshold:
                return False
        return True

    def calculate_iou(self, box1, box2):
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection

        return intersection / union if union != 0 else 0

    def show_summary(self):
        summary_text = "\n".join(f"{cls}: {count}" for cls, count in self.class_counts.items())
        summary_window = tk.Toplevel(self.root)
        summary_window.title("Detection Summary")
        summary_window.geometry("300x400")
        ttk.Label(summary_window, text="Object Detection Summary", font=("Helvetica", 14, "bold")).pack(pady=10)
        text_box = tk.Text(summary_window, wrap="word", font=("Courier", 11))
        text_box.insert(tk.END, summary_text)
        text_box.config(state=tk.DISABLED)
        text_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

        # Bring to front
        summary_window.lift()
        summary_window.attributes("-topmost", True)
        summary_window.after_idle(summary_window.attributes, "-topmost", False)

    def export_summary(self):
        if not self.class_counts:
            self.status_label.config(text="⚠️ No data to export", foreground="red")
            return
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if file_path:
            with open(file_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Class', 'Count'])
                for cls, count in self.class_counts.items():
                    writer.writerow([cls, count])
            self.status_label.config(text=f"Summary saved to {file_path}", foreground="green")

if __name__ == "__main__":
    root = tk.Tk()
    app = TrafficAnalyzerApp(root)
    root.mainloop()
