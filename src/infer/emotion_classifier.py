from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights

from src.utils.logging_config import logging

logger = logging.getLogger(__name__)


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        ## Load the pretrained model
        logger.info("Loaded the resnet50 pretrained model")
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

        ## Modify the classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        logger.info("created resnet18 with %d classes", num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
