import tkinter as tk
from tkinter import ttk

def show_summary_popup(parent, class_counts):
    summary_text = "\n".join(f"{cls}: {count}" for cls, count in class_counts.items())
    summary_window = tk.Toplevel(parent)
    summary_window.title("Detection Summary")
    summary_window.geometry("300x400")
    ttk.Label(summary_window, text="Object Detection Summary", font=("Helvetica", 14, "bold")).pack(pady=10)
    text_box = tk.Text(summary_window, wrap="word", font=("Courier", 11))
    text_box.insert(tk.END, summary_text)
    text_box.config(state=tk.DISABLED)
    text_box.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    summary_window.lift()
    summary_window.attributes("-topmost", True)
    summary_window.after_idle(summary_window.attributes, "-topmost", False)
