import torch
import torch.nn as nn

from config import (
    DROPOUT_PROBABILITY
)


class DefectClassifierCNN(nn.Module):

    def __init__(self):

        super().__init__()

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1
            ),

            nn.ReLU(),

            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                128 * 28 * 28,
                256
            ),

            nn.ReLU(),

            nn.Dropout(DROPOUT_PROBABILITY),

            nn.Linear(
                256,
                6
            )
        )

    def forward(self, x):

        x = self.features(x)

        x = self.classifier(x)

        return x