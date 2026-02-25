import cv2
import numpy as np
import time
import threading
import winsound
import urllib.request
import os

from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from mediapipe import Image, ImageFormat

MODEL_PATH = "face_landmarker.task"
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

if not os.path.exists(MODEL_PATH):
    print("face landmark model)...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("model downloaded")

def play_alert():
    try:
        winsound.Beep(1000, 600)
    except Exception:
        print("\a", end="", flush=True)

LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

EAR_THRESHOLD  = 0.25
CLOSED_SECONDS = 0

def eye_aspect_ratio(landmarks, eye_indices, img_w, img_h):
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in eye_indices]
    v1 = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
    v2 = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
    h  = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
    return (v1 + v2) / (2.0 * h) if h != 0 else 0.0

options = vision.FaceLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
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
        print("no webcam")
        return

    print("ewan ko sayo")

    eyes_closed_start = None
    alert_playing     = False
    frame_count       = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("no frame")
            break

        frame_count += 1
        img_h, img_w = frame.shape[:2]

        timestamp_ms = int(time.time() * 1000)

        rgb    = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = Image(image_format=ImageFormat.SRGB, data=rgb)
        result = detector.detect_for_video(mp_img, timestamp_ms)

        status_text  = "no face in web"
        status_color = (128, 128, 128)
        ear_avg      = 0.0
        eyes_closed  = False

        if result.face_landmarks:
            lms = result.face_landmarks[0]

            left_ear    = eye_aspect_ratio(lms, LEFT_EYE,  img_w, img_h)
            right_ear   = eye_aspect_ratio(lms, RIGHT_EYE, img_w, img_h)
            ear_avg     = (left_ear + right_ear) / 2.0
            eyes_closed = ear_avg < EAR_THRESHOLD

            if eyes_closed:
                if eyes_closed_start is None:
                    eyes_closed_start = time.time()

                elapsed = time.time() - eyes_closed_start

                if elapsed >= CLOSED_SECONDS:
                    status_text  = f"eyes closed"
                    status_color = (0, 0, 255)

                    if not alert_playing:
                        alert_playing = True
                        def _beep():
                            play_alert()
                        threading.Thread(target=_beep, daemon=True).start()
                        alert_playing = False
                else:
                    status_text  = f"eyes closed... ({elapsed:.1f}s)"
                    status_color = (0, 165, 255)
            else:
                eyes_closed_start = None
                alert_playing     = False
                status_text       = "eyes opened"
                status_color      = (0, 210, 0)

            cv2.putText(frame, f"ear: {ear_avg:.2f}",
                        (img_w - 160, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2)

        bar = frame.copy()
        cv2.rectangle(bar, (0, 0), (img_w, 55), (0, 0, 0), -1)
        cv2.addWeighted(bar, 0.45, frame, 0.55, 0, frame)
        cv2.putText(frame, status_text, (10, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, status_color, 2)

        if (eyes_closed and eyes_closed_start is not None
                and (time.time() - eyes_closed_start) >= CLOSED_SECONDS):
            cv2.rectangle(frame, (0, 0), (img_w - 1, img_h - 1), (0, 0, 255), 10)

        cv2.imshow("daming ebas", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    print("stopped")


if __name__ == "__main__":
    main()
