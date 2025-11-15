import av
import cv2
from retinaface import RetinaFace
from src.infer.predict import Prediction

model_path = "src/models/emotion-detect.pth"
emotion_model = Prediction(model_path=model_path)


def process_real_time(frame: av.VideoFrame):
    print("I see you")

    return frame


def process_video(video_path):
    video_file = cv2.VideoCapture(video_path)

    video_fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    outputs = cv2.VideoWriter(
        "/modified.mp4", fourcc=int(video_fourcc), fps=20, frameSize=(640, 480)
    )

    while True:
        ret, frame = video_file.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))

        faces = RetinaFace.detect_faces(frame, 0.5)

        for _key, value in faces.items():
            xmin, ymin, xmax, ymax = value["facial_area"]

            copy_image = frame.copy()
            cropped_image = copy_image[ymin:ymax, xmin:xmax]

            emotion_res = emotion_model.inference_emotion(cropped_image)
