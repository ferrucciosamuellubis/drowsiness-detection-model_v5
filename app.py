import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av
import cv2
import numpy as np

# --- Konfigurasi alarm ---
ALARM_AUDIO_PATH = "alarm.mp3"  # pastikan file alarm.mp3 ada di folder yang sama

# --- Streamlit UI ---
st.title("🔴 Drowsiness Detection (WebRTC)")

# Tombol play alarm (preload audio)
alarm_audio = None
with open(ALARM_AUDIO_PATH, "rb") as f:
    alarm_audio = f.read()

# --- Processor untuk deteksi kantuk ---
class DrowsinessProcessor(VideoProcessorBase):
    def __init__(self):
        # load cascades yang path-nya pasti ada
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml"
        )
        # load model TensorFlow / Keras
        from tensorflow.keras.models import load_model
        self.model = load_model("Model.h5")
        
        # state
        self.score = 0
        self.alerted = False

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            # Crop area wajah
            roi_gray = gray[y : y + h, x : x + w]
            eyes = self.eye_cascade.detectMultiScale(roi_gray, 1.1, 3)
            
            # Gambar kotak wajah
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Proses tiap mata
            for (ex, ey, ew, eh) in eyes:
                ex_abs, ey_abs = x + ex, y + ey
                eye_img = img[ey_abs : ey_abs + eh, ex_abs : ex_abs + ew]
                
                # Pre‑processing model
                eye = cv2.resize(eye_img, (80, 80))
                eye = eye / 255.0
                eye = np.expand_dims(eye, axis=0)
                
                # Prediksi
                pred = self.model.predict(eye, verbose=0)[0]
                closed_prob, open_prob = pred[0], pred[1]
                
                if closed_prob > 0.30:
                    self.score += 1
                else:
                    self.score = max(0, self.score - 1)
                
                # Gambar overlay status
                status = "Closed" if closed_prob > open_prob else "Open"
                cv2.putText(
                    img,
                    f"{status} | Score: {self.score}",
                    (10, img.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0) if status == "Open" else (0, 0, 255),
                    2,
                )
                
                # Jika score melewati threshold dan belum di‑alert
                if self.score > 15 and not self.alerted:
                    # Trigger alarm di UI
                    st.session_state["play_alarm"] = True
                    self.alerted = True
                elif self.score <= 15 and self.alerted:
                    # Reset alert jika sudah bangun
                    st.session_state["play_alarm"] = False
                    self.alerted = False

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# Inisialisasi state untuk alarm
if "play_alarm" not in st.session_state:
    st.session_state["play_alarm"] = False

# Jalankan streamer WebRTC
webrtc_ctx = webrtc_streamer(
    key="drowsiness",
    video_processor_factory=DrowsinessProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

# Setelah menjalankan frame, jika terdeteksi kantuk -> putar alarm
if st.session_state["play_alarm"]:
    st.audio(alarm_audio, format="audio/mp3", start_time=0)
