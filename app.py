# app.py

import streamlit as st
import cv2
import numpy as np
from utils import detect_emotion

st.set_page_config(page_title="Facial Expression Detector", layout="centered")
st.title("🧠 Facial Expression Recognition")
st.write("Upload an image or use your webcam to detect facial expressions!")

# Image upload
image_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])

# Camera capture
use_webcam = st.checkbox("Use Webcam")

if image_file is not None:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    results = detect_emotion(img)

    for res in results:
        x, y, w, h = res["box"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(img, f"{res['emotion']} ({res['confidence']*100:.1f}%)", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Detected Image")

elif use_webcam:
    st.warning("Click 'Start' to use webcam")

    run = st.button("Start")

    if run:
        cam = cv2.VideoCapture(0)

        stframe = st.empty()

        while True:
            ret, frame = cam.read()
            if not ret:
                break

            results = detect_emotion(frame)
            for res in results:
                x, y, w, h = res["box"]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.putText(frame, f"{res['emotion']}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

        cam.release()
