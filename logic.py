import cv2
import time
from PIL import Image, ImageTk
from collections import defaultdict
from ultralytics import YOLO
from utils import calculate_iou, is_new_detection

class VideoProcessor:
    def __init__(self, gui):
        self.gui = gui
        self.model = YOLO("yolov5s.pt")
        self.class_counts = defaultdict(int)
        self.detected_boxes = defaultdict(list)
        self.video_path = ""
        self.cap = None
        self.paused = False
        self.cancelled = False
        self.processing = False

    def prepare(self, video_path):
        self.video_path = video_path
        self.class_counts.clear()
        self.detected_boxes.clear()
        self.paused = False
        self.cancelled = False
        self.processing = True
        self.gui.status_label.config(text=f"Loading {video_path.split('/')[-1]}", foreground="blue")
        self.gui.pause_button.config(state="normal")
        self.gui.stop_button.config(state="normal")

    def toggle_pause(self):
        if not self.processing:
            return
        self.paused = not self.paused
        self.gui.pause_button.config(text="Resume" if self.paused else "Pause")

    def cancel(self):
        self.cancelled = True
        self.gui.status_label.config(text="❌ Processing cancelled", foreground="orange")
        self.gui.pause_button.config(state="disabled")
        self.gui.stop_button.config(state="disabled")

    def run(self):
        self.cap = cv2.VideoCapture(self.video_path)
        max_width, max_height = 700, 400

        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        duration_sec = total_frames / fps if fps > 0 else 0

        self.gui.progressbar["maximum"] = total_frames
        frame_number = 0
        start_time = time.time()

        while self.cap.isOpened():
            if self.cancelled:
                break
            if self.paused:
                self.gui.root.update()
                continue

            ret, frame = self.cap.read()
            if not ret:
                break

            frame_number += 1
            self.gui.progressbar["value"] = frame_number
            self.gui.status_label.config(
                text=f"Processing frame {frame_number}/{total_frames} ({(frame_number/total_frames)*100:.1f}%)"
            )
            self.gui.root.update_idletasks()

            results = self.model(frame)[0]

            for box in results.boxes:
                cls_id = int(box.cls[0].item())
                class_name = self.model.names[cls_id]
                coords = box.xyxy[0].tolist()

                if is_new_detection(class_name, coords, self.detected_boxes):
                    self.class_counts[class_name] += 1
                    self.detected_boxes[class_name].append(coords)

            annotated = results.plot()
            frame_h, frame_w = annotated.shape[:2]
            scale = min(max_width / frame_w, max_height / frame_h)
            new_w, new_h = int(frame_w * scale), int(frame_h * scale)
            resized = cv2.resize(annotated, (new_w, new_h))

            img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(img)
            self.gui.canvas.delete("all")
            self.gui.canvas.config(width=max_width, height=max_height)
            x_off = (max_width - new_w) // 2
            y_off = (max_height - new_h) // 2
            self.gui.canvas.create_image(x_off, y_off, anchor="nw", image=img_tk)
            self.gui.canvas.image = img_tk

        self.cap.release()
        self.processing = False
        self.gui.pause_button.config(state="disabled")
        self.gui.stop_button.config(state="disabled")

        if not self.cancelled:
            elapsed = time.time() - start_time
            self.gui.status_label.config(
                text=f"✅ Done: {frame_number} frames in {elapsed:.2f}s (video: {duration_sec:.2f}s)",
                foreground="green"
            )
            self.gui.show_summary()
