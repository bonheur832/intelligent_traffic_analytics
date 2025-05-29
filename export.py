from tkinter import filedialog

def export_summary_to_csv(class_counts, status_label):
    if not class_counts:
        status_label.config(text="⚠️ No data to export", foreground="red")
        return
    file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
    if file_path:
        import csv
        with open(file_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class', 'Count'])
            for cls, count in class_counts.items():
                writer.writerow([cls, count])
        status_label.config(text=f"Summary saved to {file_path}", foreground="green")
