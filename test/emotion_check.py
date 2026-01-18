import cv2
import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from src.infer.predict import Prediction

import logging

# Add configuration for both
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %I:%M:%S %p",
    handlers=[
        logging.FileHandler("log-file.log"),
        logging.StreamHandler(),
    ],
)


model_path = "src/models/emotion-detect.pth"
pred = Prediction(model_path)

image_path = "test/assets/face.jpg"
cv_image = cv2.imread(image_path)
pil_image = Image.open(image_path).convert("RGB")

logging.warning("Opencv image")
print(pred.inference_emotion(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))

print("\n\nPIL image:")
print(pred.inference_emotion(pil_image))
