import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# --- Constants ---
ALARM_AUDIO_PATH = "alarm.mp3"  # Letakkan file alarm.mp3 di direktori yang sama
MODEL_PATH = "Model.h5"        # Letakkan file Model.h5 di direktori yang sama

# --- Load alarm audio ---
try:
    with open(ALARM_AUDIO_PATH, "rb") as f:
        alarm_audio = f.read()
except FileNotFoundError:
    st.error(f"❌ Alarm audio file '{ALARM_AUDIO_PATH}' tidak ditemukan.")
    alarm_audio = None

# --- Streamlit App Title ---
st.title("🔴 Drowsiness Detection (WebRTC)")

# --- Video Processor Class ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        # Load cascades
        face_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        eye_path = cv2.data.haarcascades + "haarcascade_eye.xml"
        self.face_cascade = cv2.CascadeClassifier(face_path)
        self.eye_cascade = cv2.CascadeClassifier(eye_path)

        # Debug: verify cascades loaded
        if self.face_cascade.empty():
            st.error("❌ Gagal memuat face cascade dari: " + face_path)
        if self.eye_cascade.empty():
            st.error("❌ Gagal memuat eye cascade dari: " + eye_path)

        # Load ML model
        try:
            self.model = load_model(MODEL_PATH)
        except Exception as e:
            st.error(f"❌ Gagal memuat model: {e}")
            self.model = None

        # Initialize state
        self.score = 0
        self.alerted = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)
        for (x, y, w, h) in faces:
            # Draw face rectangle
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            roi_gray = gray[y:y+h, x:x+w]

            # Detect eyes in face region
            eyes = self.eye_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=3)
            for (ex, ey, ew, eh) in eyes:
                ex_abs, ey_abs = x+ex, y+ey
                # Draw eye rectangle
                cv2.rectangle(img, (ex_abs, ey_abs), (ex_abs+ew, ey_abs+eh), (255, 0, 0), 1)

                # Prepare eye for model
                eye_img = img[ey_abs:ey_abs+eh, ex_abs:ex_abs+ew]
                eye_resized = cv2.resize(eye_img, (80, 80))
                eye_norm = eye_resized / 255.0
                eye_input = np.expand_dims(eye_norm, axis=0)

                # Predict drowsiness
                if self.model:
                    pred = self.model.predict(eye_input, verbose=0)[0]
                    closed_prob, open_prob = pred[0], pred[1]
                else:
                    closed_prob, open_prob = 0, 1

                # Update score
                if closed_prob > open_prob:
                    self.score += 1
                else:
                    self.score = max(0, self.score - 1)

                # Overlay status text
                status_text = f"Score: {self.score}"
                cv2.putText(img, status_text, (10, img.shape[0] - 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 0, 255) if closed_prob > open_prob else (0, 255, 0), 2)

                # Trigger alarm if threshold exceeded
                if self.score > 15 and not self.alerted:
                    st.session_state.play_alarm = True
                    self.alerted = True
                elif self.score <= 15 and self.alerted:
                    st.session_state.play_alarm = False
                    self.alerted = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Initialize session state for alarm ---
if "play_alarm" not in st.session_state:
    st.session_state.play_alarm = False

# --- WebRTC streamer ---
webrtc_ctx = webrtc_streamer(
    key="drowsiness",
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# --- Play alarm if triggered ---
if st.session_state.play_alarm and alarm_audio:
    st.audio(alarm_audio, format="audio/mp3", start_time=0)
