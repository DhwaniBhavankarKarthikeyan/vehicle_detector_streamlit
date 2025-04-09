import streamlit as st
import tempfile
import os
from utils.audio_processing import extract_audio, compute_energy_envelope
from utils.gan_model import enhance_audio_with_gan, get_gan_model
from utils.detection import detect_vehicle_times
from utils.video_utils import detect_vehicles_in_video, play_video

st.title("🚗 Vehicle Entry Detection from Video using Audio-GAN")

uploaded_file = st.file_uploader("Upload a video", type=["mp4"])
if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded_file.read())
        video_path = tmp.name

    st.video(video_path)

    st.write("⏳ Extracting audio...")
    audio_path = extract_audio(video_path)

    st.write("🎛️ Enhancing audio using GAN...")
    enhanced_audio = enhance_audio_with_gan(audio_path)

    st.write("📈 Analyzing energy envelope...")
    energy, suspicious_times = compute_energy_envelope(enhanced_audio)

    st.success(f"🚨 Likely vehicle entries (in seconds): {suspicious_times[:5]}")

    st.write("🚙 Running object detection...")
    detection_video_path = detect_vehicles_in_video(video_path, suspicious_times)

    st.video(detection_video_path)
