from flask import Blueprint, render_template, Response, jsonify, request
import cv2
import numpy as np
from keras.models import load_model
import os
from collections import deque

violence_bp = Blueprint('violence', __name__)

class ViolenceDetector:
    def __init__(self, model_path="violence.h5", frame_buffer_size=16):
        self.model = load_model(model_path)
        self.frame_buffer_size = frame_buffer_size
        self.frame_buffer = deque(maxlen=frame_buffer_size)

    def preprocess_frame(self, frame):
        try:
            frame = cv2.resize(frame, (64, 64))
            frame = np.array(frame) / 255.0
            return frame
        except Exception as e:
            print(f"Error preprocessing frame: {str(e)}")
            return None

    def predict_violence(self, frames):
        try:
            if len(frames) != self.frame_buffer_size:
                return None, None

            frames_array = np.array(frames)
            frames_array = np.expand_dims(frames_array, axis=0)
            prediction = self.model.predict(frames_array, verbose=0)
            predicted_class = np.argmax(prediction)

            return predicted_class, prediction[0]
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            return None, None


detector = ViolenceDetector()


def process_frame(frame):
    processed_frame = detector.preprocess_frame(frame)
    if processed_frame is not None:
        detector.frame_buffer.append(processed_frame)

    if len(detector.frame_buffer) == detector.frame_buffer_size:
        predicted_class, probabilities = detector.predict_violence(list(detector.frame_buffer))
        return predicted_class, probabilities
    return None, None


@violence_bp.route('/')
def index():
    return render_template('detection.html')


@violence_bp.route('/upload', methods=['POST'])
def upload_video():
    if 'video' not in request.files:
        return jsonify({'error': 'No video file uploaded'}), 400

    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Save the uploaded file temporarily
    temp_path = os.path.join('temp', video_file.filename)
    video_file.save(temp_path)

    try:
        cap = cv2.VideoCapture(temp_path)
        results = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            predicted_class, probabilities = process_frame(frame)
            if predicted_class is not None:
                results.append({
                    'class': int(predicted_class),
                    'probability': float(probabilities[predicted_class])
                })

        cap.release()
        os.remove(temp_path)  # Clean up temporary file

        # Calculate overall results
        if results:
            violence_frames = sum(1 for r in results if r['class'] == 1)
            violence_probability = sum(r['probability'] for r in results if r['class'] == 1) / len(results)

            return jsonify({
                'violence_detected': violence_frames > len(results) * 0.3,
                'probability': violence_probability,
                'frame_results': results
            })

        return jsonify({'error': 'No results generated'}), 500

    except Exception as e:
        return jsonify({'error': str(e)}), 500


def gen_frames():
    camera = cv2.VideoCapture(0)
    try:
        while True:
            success, frame = camera.read()
            if not success:
                break

            predicted_class, probabilities = process_frame(frame)

            if predicted_class is not None:
                color = (0, 0, 255) if predicted_class == 1 else (0, 255, 0)
                status = "VIOLENCE DETECTED!" if predicted_class == 1 else "No Violence"
                cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, color, 2)

                if probabilities is not None:
                    prob_text = f"Confidence: {probabilities[predicted_class]:.2f}"
                    cv2.putText(frame, prob_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, color, 2)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    finally:
        camera.release()


@violence_bp.route('/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')