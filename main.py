import cv2
import numpy as np
import time
import threading
import winsound
import urllib.request
import os

import tensorflow as tf
from tensorflow import keras

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

LANDMARK_MODEL_PATH = "face_landmarker.task"
LANDMARK_MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)
KERAS_MODEL_PATH = "eye_state_model.keras"

if not os.path.exists(LANDMARK_MODEL_PATH):
    print("Downloading face landmark model...")
    urllib.request.urlretrieve(LANDMARK_MODEL_URL, LANDMARK_MODEL_PATH)
    print("Landmark model downloaded.")

print("Loading Keras eye model...")
eye_model = keras.models.load_model(KERAS_MODEL_PATH)
EYE_IMG_SIZE = (24, 24)
CLOSED_CONFIDENCE = 0.5
print("Keras model loaded.")

def play_alert():
    try:
        winsound.Beep(1000, 600)
    except Exception:
        print("\a", end="", flush=True)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

LEFT_EYE_CORNERS  = [362, 263]
RIGHT_EYE_CORNERS = [33,  133]

CLOSED_SECONDS = 2

def crop_eye(frame, landmarks, corner_indices, img_w, img_h, pad=10):
    xs = [landmarks[i].x * img_w for i in corner_indices]
    ys = [landmarks[i].y * img_h for i in corner_indices]
    x1, x2 = int(min(xs)) - pad, int(max(xs)) + pad
    y1, y2 = int(min(ys)) - pad, int(max(ys)) + pad
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, img_w), min(y2, img_h)
    return frame[y1:y2, x1:x2]

def preprocess_eye(eye_crop):
    gray = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, EYE_IMG_SIZE)
    normalized = resized.astype("float32") / 255.0
    return normalized.reshape(1, EYE_IMG_SIZE[0], EYE_IMG_SIZE[1], 1)

def predict_eye_open(eye_crop):
    if eye_crop is None or eye_crop.size == 0:
        return True
    tensor = preprocess_eye(eye_crop)
    prob_open = float(eye_model.predict(tensor, verbose=0)[0][0])
    return prob_open >= CLOSED_CONFIDENCE

options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=LANDMARK_MODEL_PATH),
    running_mode=vision.RunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
detector = vision.FaceLandmarker.create_from_options(options)

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("No webcam found.")
        return

    print("Running — press Q to quit.")

    eyes_closed_start = None
    alert_playing     = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("No frame received.")
            break

        img_h, img_w = frame.shape[:2]
        timestamp_ms = int(time.time() * 1000)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, timestamp_ms)

        status_text  = "No face detected"
        status_color = (128, 128, 128)
        eyes_closed  = False

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            left_crop  = crop_eye(frame, lms, LEFT_EYE_CORNERS,  img_w, img_h)
            right_crop = crop_eye(frame, lms, RIGHT_EYE_CORNERS, img_w, img_h)

            left_open  = predict_eye_open(left_crop)
            right_open = predict_eye_open(right_crop)

            eyes_closed = not (left_open and right_open)

            if eyes_closed:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()

                elapsed = time.time() - eyes_closed_start

                if elapsed >= CLOSED_SECONDS:
                    status_text  = "EYES CLOSED — WAKE UP!"
                    status_color = (0, 0, 255)

                    if not alert_playing:
                        alert_playing = True
                        def _beep():
                            play_alert()
                        threading.Thread(target=_beep, daemon=True).start()
                        alert_playing = False
                else:
                    status_text  = f"Eyes closing... ({elapsed:.1f}s)"
                    status_color = (0, 165, 255)
            else:
                eyes_closed_start = None
                alert_playing     = False
                status_text       = "Eyes open"
                status_color      = (0, 210, 0)

            cv2.putText(frame, f"L: {'open' if left_open else 'closed'}",
                        (img_w - 160, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            cv2.putText(frame, f"R: {'open' if right_open else 'closed'}",
                        (img_w - 160, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)

        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (img_w, 55), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, status_text, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

        if eyes_closed and eyes_closed_start is not None and \
                (time.time() - eyes_closed_start) >= CLOSED_SECONDS:
            cv2.rectangle(frame, (0, 0), (img_w - 1, img_h - 1), (0, 0, 255), 10)

        cv2.imshow("Drowsiness Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("Stopped.")

if __name__ == "__main__":
    main()
