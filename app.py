from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import cv2
import base64
import numpy as np
from exercise_tracker import ExerciseTracker

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

tracker = ExerciseTracker()

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('video_frame')
def handle_video_frame(data):
    # Decode base64 image
    encoded_data = data.split(',')[1]
    nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Resize frame if it's too large
    max_width = 640
    if frame.shape[1] > max_width:
        scale = max_width / frame.shape[1]
        frame = cv2.resize(frame, None, fx=scale, fy=scale)

    # Process frame with ExerciseTracker
    processed_frame = tracker.process_frame(frame)

    # Encode processed frame to base64
    _, buffer = cv2.imencode('.jpg', processed_frame)
    processed_frame_base64 = base64.b64encode(buffer).decode('utf-8')

    # Get exercise state
    exercise_state = tracker.get_exercise_state()

    # Emit processed frame and exercise state
    emit('processed_frame', {'image': processed_frame_base64, 'state': exercise_state})

if __name__ == '__main__':
    socketio.run(app, debug=True)