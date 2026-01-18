import sys
import logging
import streamlit as st
from streamlit_webrtc import webrtc_streamer
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from src.utils.logging_config import configure_logging
configure_logging()

from src.utils.process_frame import process_real_time, process_video


# Basic logging configuration: file + console (in separate module)
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
    uploaded_file = st.file_uploader("Video file", type=["mp4", "avi", "mov", "mkv"])

    if st.button("Infer") and uploaded_file:
        video_path = "/tmp/input.mp4"
        video_bytes = uploaded_file.read()

        with open(video_path, "wb") as f:
            f.write(video_bytes)

        st.write("### Processing Video")
        # Create a placeholder for video frames
        video_placeholder = st.empty()

        with st.spinner("Processing the video...", show_time=True):
            output_path = process_video(video_path=video_path, placeholder=video_placeholder)

        if output_path:
            st.success("Video processing completed!")
            # Provide download button
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="emotion_detected.mp4",
                    mime="video/mp4"
                )
