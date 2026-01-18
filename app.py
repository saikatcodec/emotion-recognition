import sys
import logging
from pathlib import Path
from src.utils.logging_config import configure_logging
import streamlit as st
from streamlit_webrtc import webrtc_streamer

from src.utils.process_frame import process_real_time, process_video

sys.path.append(str(Path(__file__).parent))

# Basic logging configuration: file + console (in separate module)
configure_logging()
logger = logging.getLogger(__name__)

st.title("Face Emotion Recognition")
st.subheader("Recognize the face emotion to video", divider="gray")

real_time = st.checkbox("Real-time process(Camera)")
if real_time:
    st.write("Real-time process")
    webrtc_streamer(
        key="emotion-detect-camera-access",
        sendback_audio=False,
        video_frame_callback=process_real_time,
    )
else:
    # video_path = st.text_input("Video path", placeholder="path/to/video.mp4")
    uploaded_file = st.file_uploader("Video file", type=["mp4", "avi", "mov"])

    if st.button("Infer") and uploaded_file:
        video_path = "/tmp/input.mp4"
        video_bytes = uploaded_file.read()

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        with st.spinner("Processing the video...", show_time=True):
            output_path = process_video(video_path=video_path)

        st.video(output_path)
