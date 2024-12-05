from flask import Flask, render_template, Response, jsonify, request
from flask_socketio import SocketIO, emit
import cv2
import os
import logging
import base64
import numpy as np
from virtual_tryon import detect_eyes_and_forehead, overlay_clothing
import cvzone
from cvzone.PoseModule import PoseDetector

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet')  # WebSocket support

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load accessories
hat_folder = "data/hat"
glasses_folder = "data/glasses"
shirt_folder = "data/Shirts"

hat_files = sorted([f for f in os.listdir(hat_folder) if f.endswith(('png', 'jpeg', 'jpg'))])
glasses_files = sorted([f for f in os.listdir(glasses_folder) if f.endswith(('png', 'jpeg', 'jpg'))])
shirt_files = sorted([f for f in os.listdir(shirt_folder) if f.endswith(('png', 'jpeg', 'jpg'))])

if not hat_files or not glasses_files or not shirt_files:
    raise FileNotFoundError("Ensure hat, glasses, and shirt images exist at the specified paths.")

hats = [cv2.imread(os.path.join(hat_folder, f), cv2.IMREAD_UNCHANGED) for f in hat_files]
glasses_list = [cv2.imread(os.path.join(glasses_folder, f), cv2.IMREAD_UNCHANGED) for f in glasses_files]
shirts = [cv2.imread(os.path.join(shirt_folder, f), cv2.IMREAD_UNCHANGED) for f in shirt_files]

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    logger.error("Cannot access camera. Ensure the camera is connected and accessible.")
    cap = None  # Prevent further use of the camera

# Initialize variables for selecting accessories
hat_index = 0
glasses_index = 0
shirt_index = 0


def overlay_shirt(frame, lmList, bboxInfo, shirt):
    """Overlay shirt on detected person using pose keypoints."""
    if bboxInfo:
        bbox = bboxInfo["bbox"]
        center = bboxInfo["center"]

        shirt_width = int(bbox[2] * 0.8)  # 80% of person's width
        shirt_height = int(shirt_width * 581 / 440)  # Maintain aspect ratio

        x1 = max(0, center[0] - shirt_width // 2)
        y1 = max(0, bbox[1] + bbox[3] // 6)  # Start from 1/6th of the body height

        # Overlay the shirt image onto the frame
        shirt_resized = cv2.resize(shirt, (shirt_width, shirt_height))
        frame = cvzone.overlayPNG(frame, shirt_resized, (x1, y1))
    return frame


@app.route('/')
def index():
    """Render home page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Stream video feed."""
    def generate_frames():
        global cap, hat_index, glasses_index, shirt_index
        if not cap:
            while True:
                frame = cv2.imread("data/no_camera_placeholder.jpg")  # Placeholder image if no camera is available
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
        else:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Flip frame horizontally for mirror effect
                frame = cv2.flip(frame, 1)

                # Detect pose
                frame = detector.findPose(frame, draw=False)
                lmList, bboxInfo = detector.findPosition(frame, bboxWithHands=False, draw=False)

                # Detect eyes and forehead
                eyes, forehead = detect_eyes_and_forehead(frame, face_cascade, eye_cascade)

                # Overlay accessories (glasses, hat, shirt)
                frame = overlay_clothing(frame, eyes, forehead, hats[hat_index], glasses_list[glasses_index])
                frame = overlay_shirt(frame, lmList, bboxInfo, shirts[shirt_index])

                # Encode frame in JPEG for streaming
                ret, buffer = cv2.imencode('.jpg', frame)
                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/change_hat', methods=['POST'])
def change_hat():
    """Change hat accessory."""
    global hat_index
    hat_index = (hat_index + 1) % len(hats)
    return jsonify({'status': 'success'})


@app.route('/change_glasses', methods=['POST'])
def change_glasses():
    """Change glasses accessory."""
    global glasses_index
    glasses_index = (glasses_index + 1) % len(glasses_list)
    return jsonify({'status': 'success'})


@app.route('/change_shirt', methods=['POST'])
def change_shirt():
    """Change shirt accessory."""
    global shirt_index
    shirt_index = (shirt_index + 1) % len(shirts)
    return jsonify({'status': 'success'})


@app.route('/process_frame', methods=['POST'])
def process_frame():
    """Process frames sent by the browser."""
    try:
        data = request.json
        image_data = data['image']

        # Decode the base64-encoded image
        image_data = base64.b64decode(image_data.split(',')[1])
        np_image = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(np_image, cv2.IMREAD_COLOR)

        # Process the frame (e.g., add virtual accessories)
        processed_frame = frame  # Placeholder: Add your processing logic here

        # Optionally, encode the processed frame and return it
        _, buffer = cv2.imencode('.jpg', processed_frame)
        processed_image_base64 = base64.b64encode(buffer).decode('utf-8')

        return jsonify({'status': 'success', 'image': processed_image_base64})
    except Exception as e:
        logger.error(f"Error processing frame: {str(e)}")
        return jsonify({'status': 'error', 'message': str(e)})


# WebSocket signaling for WebRTC
@socketio.on('offer')
def handle_offer(data):
    emit('offer', data, broadcast=True)


@socketio.on('answer')
def handle_answer(data):
    emit('answer', data, broadcast=True)


@socketio.on('candidate')
def handle_candidate(data):
    emit('candidate', data, broadcast=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=True)
