from flask import Flask, render_template, Response, request, redirect, url_for
import torch
import cv2
import os
from collections import defaultdict
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5s model from local directory
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')

video_path = None  # Global path to uploaded video
final_counts = defaultdict(int)  # Store total class counts

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def index():
    global video_path
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            video_path = filepath
            return redirect(url_for('index', video_feed=True))
    return render_template('index.html', video_feed=bool(video_path))

def generate_frames(path):
    global final_counts
    final_counts.clear()  # Reset counts
    cap = cv2.VideoCapture(path)

    while True:
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        df = results.pandas().xyxy[0]  # DataFrame with detections
        frame_counts = df['name'].value_counts().to_dict()

        for cls, count in frame_counts.items():
            final_counts[cls] += count

        annotated_frame = results.render()[0].copy()
        # Optional: display current frame counts
        y_offset = 30
        for i, (cls, count) in enumerate(frame_counts.items()):
            text = f"{cls}: {count}"
            cv2.putText(annotated_frame, text, (10, y_offset + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/video_feed')
def video_feed():
    if not video_path:
        return "No video uploaded"
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/results')
def results():
    return render_template('results.html', class_counts=dict(final_counts))

if __name__ == '__main__':
    app.run(debug=True)
