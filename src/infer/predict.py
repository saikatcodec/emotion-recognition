import torch
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision.models.resnet import ResNet18_Weights
from src.infer.emotion_classifier import ResNet


class Prediction:
    def __init__(self, model_path):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.model_path = model_path

        ## Load saved model informations
        checkpoint = torch.load(self.model_path, map_location=self.device)
        self.classes = checkpoint["classes"]
        model_dict = checkpoint["model_state_dict"]

        ## Load the saved model
        self.model = ResNet(num_classes=len(self.classes)).to(self.device)
        self.model.load_state_dict(model_dict)
        self.model.eval()

        ## Load pretrained transforms
        self.transforms = ResNet18_Weights.IMAGENET1K_V1.transforms()

    def inference_emotion(self, image):
        image = T.ToTensor()(image)
        image = self.transforms(image).to(self.device)
        image_batch = image.unsqueeze(0)

        with torch.no_grad():
            outputs = self.model(image_batch)
        probabilites = F.softmax(outputs[0], dim=0)
        values, ids = torch.topk(probabilites, k=5)

        results = {}
        for value, id in zip(values, ids):
            results[self.classes[id]] = value.cpu().item()

        return results
