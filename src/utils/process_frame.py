import av
import cv2

from retinaface import RetinaFace
from src.infer.predict import Prediction
import logging

logger = logging.getLogger(__name__)

model_path = "src/models/emotion_detect_v2.0.pth"
emotion_model = Prediction(model_path=model_path)

# Frame skipping counter for real-time processing
frame_counter = 0

def process_real_time(frame: av.VideoFrame):
    global frame_counter
    frame_counter += 1

    image = frame.to_ndarray(format="rgb24")
    image = cv2.resize(image, (480, 360))

    if frame_counter % 3 == 0:
        faces = RetinaFace.detect_faces(image, threshold=0.7) or {}
        for _, value in faces.items():
            xmin, ymin, xmax, ymax = value["facial_area"]

            cropped = image[ymin:ymax, xmin:xmax]
            emotion_res = emotion_model.inference_emotion(cropped)

            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (200, 12, 0), 2)

            for i, (name, value) in enumerate(emotion_res.items()):
                cv2.putText(
                    image,
                    f"{name}: {value:.2f}",
                    (xmin, ymax + (i + 1) * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )

    out = av.VideoFrame.from_ndarray(image, format="rgb24")
    out.pts = frame.pts
    out.time_base = frame.time_base
    return out



def process_video(video_path, placeholder=None):
    video_file = cv2.VideoCapture(video_path)
    if video_file.isOpened():
        logger.info(f"Video opened for {video_path}")
    else:
        logger.error(f"Video is not opened for {video_path}")
        return None

    video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outputs_path = "output/modified.mp4"
    outputs = cv2.VideoWriter(
        outputs_path, fourcc=int(video_fourcc), fps=20, frameSize=(640, 480)
    )
    logger.warning('Modified video file is created with name of "/modified.mp4"')

    no_frame = 0
    while True:
        ret, frame = video_file.read()

        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        no_frame += 1
        faces = RetinaFace.detect_faces(frame, 0.7)

        if not faces:
            logger.warning(f"{no_frame} no of frame has no face")
            faces = {}

        for _key, value in faces.items():
            xmin, ymin, xmax, ymax = value["facial_area"]

            copy_image = frame.copy()
            cropped_image = copy_image[ymin:ymax, xmin:xmax]

            emotion_res = emotion_model.inference_emotion(cropped_image)

            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (200, 12, 0), 2)

            for i, (name, value) in enumerate(emotion_res.items()):
                cv2.putText(
                    frame,
                    f"{name}: {value:.3f}",
                    (xmin, ymax + (i + 2) * 25),
                    cv2.FONT_HERSHEY_PLAIN,
                    2,
                    (200, 12, 0),
                    2,
                    cv2.LINE_AA,
                )

        # Display frame in real-time if placeholder is provided
        if placeholder is not None:
            placeholder.image(frame, channels="RGB", width='stretch')

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        outputs.write(frame_bgr)

    outputs.release()
    video_file.release()

    return outputs_path
