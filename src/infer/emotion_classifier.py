from torch import nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

from src.utils.logging_config import logging

logger = logging.getLogger(__name__)


class ResNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        ## Load the pretrained model
        logger.info("Loaded the resnet18 pretrained model")
        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        ## Modify the classification layer
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)
        logger.info("created resnet18 with %d classes", num_classes)

    def forward(self, x):
        x = self.model(x)
        return x
