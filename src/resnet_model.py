import torch.nn as nn
from torchvision import models


class ResNet18Transfer(nn.Module):

    def __init__(self):

        super().__init__()

        self.model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            in_features,
            6
        )


    def forward(self, x):
        return self.model(x)