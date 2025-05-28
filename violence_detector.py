import cv2
import numpy as np
from keras.models import load_model
from collections import deque

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

def process_frame(frame, detector):
    processed_frame = detector.preprocess_frame(frame)
    if processed_frame is not None:
        detector.frame_buffer.append(processed_frame)

    if len(detector.frame_buffer) == detector.frame_buffer_size:
        predicted_class, probabilities = detector.predict_violence(list(detector.frame_buffer))
        return predicted_class, probabilities
    return None, None