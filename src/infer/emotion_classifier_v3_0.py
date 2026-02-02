from torch import nn
from torchvision.models.resnet import resnet50, ResNet50_Weights

from src.utils.logging_config import logging

logger = logging.getLogger(__name__)


class ResNet(nn.Module):
  def __init__(self, num_classes):
    super().__init__()

    self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

    # Freeze first 20th the layer
    for param in list(self.model.parameters())[:10]:
      param.requires_grad = False

    # Freeze the layer
    for param in list(self.model.parameters())[50:60]:
      param.requires_grad = False

    ## Modify the classification layer
    in_features = self.model.fc.in_features
    self.model.fc = nn.Sequential(
      nn.Dropout(0.5, inplace=True),
      nn.Linear(in_features, num_classes)
    )

  def forward(self, x):
    x = self.model(x)
    return x