import cv2
import os
import numpy as np

from retinaface import RetinaFace
from src.infer.predict import Prediction
from src.utils.logging_config import logging

logger = logging.getLogger(__name__)

model_path = "src/models/emotion_detect_v3.0.pth"
emotion_model = Prediction(model_path=model_path)

def convert_to_opencv(picture):
    '''
    Used to convert image file buffer to open-cv image
    
    :param picture: image file buffer
    :return: open-cv image
    '''
    bytes_data = picture.getvalue()
    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    logger.info('Image are converted into open-cv')

    return cv2_img

def save_to_path(cv_img):
    file_path = 'outputs/camera-photo.jpg'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(file_path, cv_img)

    return file_path


def from_photos(cv_img):
    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

    faces = RetinaFace.detect_faces(cv_img, 0.7)

    if not faces:
        logger.warning(f"The photo has no face")
        faces = {}

    for _key, value in faces.items():
        xmin, ymin, xmax, ymax = value["facial_area"]

        copy_image = cv_img.copy()
        cropped_image = copy_image[ymin:ymax, xmin:xmax]

        cropped_image = cv2.resize(cropped_image, (256, 256))
        emotion_res: dict = emotion_model.inference_emotion(cropped_image)

        cv2.rectangle(cv_img, (xmin, ymin), (xmax, ymax), (200, 12, 0), 2)

        for i, (name, value) in enumerate(emotion_res.items()):
            cv2.putText(
                cv_img,
                f"{name}: {value:.3f}",
                (xmin - 10, ymax + (i + 2) * 25),
                cv2.FONT_HERSHEY_PLAIN,
                2,
                (200, 12, 0),
                2,
                cv2.LINE_AA,
            )

    return cv_img, emotion_res


def process_video(video_path, placeholder=None):
    video_file = cv2.VideoCapture(video_path)
    if video_file.isOpened():
        logger.info(f"Video opened for {video_path}")
    else:
        logger.error(f"Video is not opened for {video_path}")
        return None

    # Calculate aspect ratio
    video_w = video_file.get(cv2.CAP_PROP_FRAME_WIDTH)
    video_h = video_file.get(cv2.CAP_PROP_FRAME_HEIGHT)
    aspect_ratio = video_w * 1.0 / video_h

    # Create output video
    video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outputs_path = "outputs/modified.mp4"
    os.makedirs(os.path.dirname(outputs_path), exist_ok=True)
    outputs_h = 480
    outputs = cv2.VideoWriter(
        outputs_path, fourcc=int(video_fourcc), fps=20, frameSize=(int(aspect_ratio * outputs_h), outputs_h)
    )
    logger.warning('Modified video file is created with name of "/modified.mp4"')

    no_frame = 0
    while video_file.isOpened():
        ret = video_file.grab()
        if not ret:
            break

        no_frame += 1
        if no_frame % 5 != 0:
            continue

        ret, frame = video_file.retrieve()
        if not ret:
            break

        frame = cv2.resize(frame, (int(aspect_ratio * outputs_h), outputs_h))

        frame, _ = from_photos(frame)

        # Display frame in real-time if placeholder is provided
        if placeholder is not None:
            placeholder.image(frame, channels="RGB")

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        outputs.write(frame_bgr)

    outputs.release()
    video_file.release()

    return outputs_path
