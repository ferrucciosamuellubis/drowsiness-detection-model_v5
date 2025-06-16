import cv2
import numpy as np
import av
from tensorflow.keras.models import load_model
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit as st
import os

# Load model and cascade
model = load_model("Model.h5")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# UI
st.set_page_config(layout="centered")
st.markdown("## 🔴 Drowsiness Detection (WebRTC)")

# Audio file for alarm
ALARM_PATH = "alarm.mp3"
if not os.path.exists(ALARM_PATH):
    st.warning("alarm.mp3 file not found!")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.score = 0
        self.alert_displayed = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        eyes = eye_cascade.detectMultiScale(gray)
        for (ex, ey, ew, eh) in eyes:
            eye = img[ey:ey + eh, ex:ex + ew]
            eye = cv2.resize(eye, (80, 80))
            eye = eye / 255.0
            eye = np.expand_dims(eye, axis=0)

            pred = model.predict(eye, verbose=0)
            if pred[0][0] > 0.3:  # Closed
                self.score += 1
            else:
                self.score -= 1

            self.score = max(0, min(self.score, 30))

        # Alarm
        if self.score > 15 and not self.alert_displayed:
            self.alert_displayed = True
            os.system(f"ffplay -nodisp -autoexit {ALARM_PATH} &")
        elif self.score <= 15:
            self.alert_displayed = False

        # Show score
        cv2.putText(img, f"Score: {self.score}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
