# Face Emotion Recognition

A real-time face emotion recognition system that detects faces and classifies emotions using deep learning. Built with PyTorch, RetinaFace for face detection, and Streamlit for the web interface.

## Features

- **Real-time emotion detection** via webcam using WebRTC
- **Video file processing** with emotion overlay
- **Face detection** using RetinaFace
- **Emotion classification** using fine-tuned ResNet18
- **Web interface** built with Streamlit

## Project Structure

```
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── src/
│   ├── infer/
│   │   ├── emotion_classifier.py   # ResNet model definition
│   │   └── predict.py              # Inference class
│   ├── models/
│   │   ├── emotion_detect_v2.0.pth # Trained model weights
│   │   └── emotion-detect.pth
│   ├── train/
│   │   ├── emotion_detection_with_resnet.ipynb  # Training notebook
│   │   └── Detect_face_using_RetinaFace.ipynb   # Face detection experiments
│   └── utils/
│       ├── logging_config.py       # Logging configuration
│       └── process_frame.py        # Frame processing utilities
├── test/
│   └── emotion_check.py            # Test script
└── output/                         # Processed video output
```

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd emotion-recog
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Web Application

Run the Streamlit app:

```bash
streamlit run app.py
```

#### Real-time Mode

- Check the "Real-time process(Camera)" checkbox
- Allow camera access when prompted
- View live emotion detection on your face

#### Video File Mode

- Upload a video file (mp4, avi, mov, mkv)
- Click "Infer" to process
- Download the processed video with emotion annotations

### Programmatic Usage

```python
from src.infer.predict import Prediction

# Initialize the model
model_path = "src/models/emotion_detect_v2.0.pth"
predictor = Prediction(model_path)

# Run inference on an image (numpy array or PIL Image)
emotions = predictor.inference_emotion(image)
print(emotions)
# Output: {'happy': 0.85, 'neutral': 0.10, 'sad': 0.03, ...}
```

## Model Training

The emotion classifier is trained on facial emotion datasets using transfer learning with ResNet18. Training details can be found in [src/train/emotion_detection_with_resnet.ipynb](src/train/emotion_detection_with_resnet.ipynb).

### Training Metrics

- **Training Accuracy**: ~78%
- **Validation Accuracy**: ~55-57%
- **Epochs**: 100

## Supported Emotions

The model classifies the following emotions:

- Happy
- Sad
- Angry
- Surprise
- Neutral
- Confused

## Dependencies

Key dependencies include:

- PyTorch
- Streamlit
- OpenCV
- RetinaFace
- streamlit-webrtc
- Pillow

See [requirements.txt](requirements.txt) for full list.

## License

MIT License
