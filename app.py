from flask import Flask, render_template, Response, request, jsonify
import cv2
import numpy as np
from tensorflow import keras
from src.components import EmotionDetector
import base64
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

mt = EmotionDetector.EmotionDetector()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(mt.generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    emotion = mt.masterImageHandler(file)
    html_text = image_to_html(emotion)
    return html_text


app.config['UPLOAD_FOLDER'] = "src/components/"
@app.route('/upload_video', methods=['POST'])
def video():
    if 'video' not in request.files:
        return 'No video part'
    video = request.files['video']
    if video.filename == '':
        return 'No selected video file'
    if video:
        video.save("ML_end_to_end/src/components/videos/temp.mp4")
        print("video uploaded")
        emotion = mt.videoHandler() 
        return Response(emotion, mimetype='multipart/x-mixed-replace; boundary=frame')
    
def image_to_html(image):
    return f'<img src="data:image/jpeg;base64,{base64.b64encode(image).decode()}" height="100%"/>'

if __name__ == '__main__':
    print("\n \n", "---"*20)
    print("running at localhost:5000")
    app.run(debug=True)