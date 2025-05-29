import tkinter as tk
from tkinter import filedialog, ttk
import threading
from logic import VideoProcessor
from summary import show_summary_popup
from export import export_summary_to_csv

class TrafficAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Traffic Analyzer")
        self.root.geometry("960x760")
        self.root.configure(bg="#f0f2f5")

        self.processor = VideoProcessor(self)
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

    def select_video(self):
        video_path = filedialog.askopenfilename(
            title="Select a Video",
            filetypes=[("Video Files", "*.mp4 *.avi *.mov *.mkv")]
        )
        if video_path:
            self.processor.prepare(video_path)
            threading.Thread(target=self.processor.run, daemon=True).start()

    def toggle_pause(self):
        self.processor.toggle_pause()

    def stop_processing(self):
        self.processor.cancel()

    def export_summary(self):
        export_summary_to_csv(self.processor.class_counts, self.status_label)

    def show_summary(self):
        show_summary_popup(self.root, self.processor.class_counts)
