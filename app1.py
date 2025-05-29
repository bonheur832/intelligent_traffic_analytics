from flask import Flask, render_template, Response, request, redirect, url_for
import torch
import cv2
import os
from werkzeug.utils import secure_filename
import pandas as pd
import seaborn as sns



UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
model = torch.hub.load('yolov5', 'custom', path='yolov5s.pt', source='local')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

video_path = None  # Global path to uploaded video

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
            return redirect(url_for('video_feed'))
    return render_template('index.html')

def generate_frames(path):
    cap = cv2.VideoCapture(path)
    while True:
        success, frame = cap.read()
        if not success:
            break
        results = model(frame)
        annotated_frame = results.render()[0]
        ret, buffer = cv2.imencode('.jpg', annotated_frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if not video_path:
        return "No video uploaded"
    return Response(generate_frames(video_path), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
