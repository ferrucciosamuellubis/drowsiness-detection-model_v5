import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time
import subprocess
import platform

st.set_page_config(page_title="Drowsiness Detection", page_icon="🚗")
st.title("🚗 Drowsiness Detection System")

# Load model
@st.cache_resource
def load_drowsiness_model():
    try:
        return load_model("Model.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_drowsiness_model()

# Load cascades
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
if face_cascade.empty(): st.error("Could not load face cascade")
if eye_cascade.empty(): st.error("Could not load eye cascade")

# Alarm
def play_alarm():
    audio = "alarm.mp3"
    try:
        sys = platform.system()
        if sys == "Darwin":
            subprocess.Popen(['afplay', audio])
        elif sys == "Linux":
            try:
                subprocess.Popen(['aplay', audio])
            except FileNotFoundError:
                try: subprocess.Popen(['paplay', audio])
                except: st.warning("🔊 Alarm: DROWSINESS DETECTED!")
        elif sys == "Windows":
            import winsound
            winsound.PlaySound(audio, winsound.SND_FILENAME)
    except:
        st.warning("🔊 DROWSINESS DETECTED!")

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.score = 0
        self.last_alarm = time.time()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 3)

        for x, y, w, h in faces:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            for ex, ey, ew, eh in eyes:
                cv2.rectangle(img, (x+ex, y+ey), (x+ex+ew, y+ey+eh), (0, 255, 0), 2)
                roi = roi_gray[ey:ey+eh, ex:ex+ew]
                if roi.size > 0 and model:
                    eye = cv2.resize(roi, (80, 80))/255.0
                    eye = eye.reshape(-1,80,80,1)
                    pred = model.predict(eye, verbose=0)[0,0]
                    if pred < 0.5:
                        self.score += 1
                    else:
                        self.score = max(0, self.score -1)

        cv2.putText(img, f"Score: {self.score}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0),2)

        if self.score > 15:
            cv2.putText(img, "DROWSINESS ALERT!", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255),2)
            if time.time() - self.last_alarm > 2:
                play_alarm()
                self.last_alarm = time.time()

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# WebRTC setup
rtc_conf = RTCConfiguration({"iceServers":[{"urls":["stun:stun.l.google.com:19302"]}]})
webrtc_streamer(key="drowsiness", video_processor_factory=VideoProcessor, rtc_configuration=rtc_conf)

# Instructions
st.markdown("""
**Instructions:**
1. Izinkan akses kamera ketika diminta.
2. Pastikan wajah terlihat jelas.
3. Alarm berbunyi saat "Score" > 15.
""")
