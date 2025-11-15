import cv2
import sys
from pathlib import Path
from PIL import Image

sys.path.append(str(Path(__file__).parent.parent))

from src.infer.predict import Prediction


model_path = "src/models/emotion-detect.pth"
pred = Prediction(model_path)

image_path = "test/assets/face.jpg"
cv_image = cv2.imread(image_path)
pil_image = Image.open(image_path).convert("RGB")

print("For opencv image:")
print(pred.inference_emotion(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)))

print("\n\nPIL image:")
print(pred.inference_emotion(pil_image))
