import torch
import torch.nn as nn 
from typing import Any

class AlexNet(nn.Module):

  def __init__(self, num_classes: int = 1000) -> None:
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
        # first layer(convolutional layer)
        nn.Conv2d(in_channels=3, out_channels=96, stride=4, padding=2),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=5, k=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # second layer(convolutional layer)
        nn.Conv2d(in_channels=96, out_channels=256, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=5, k=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        # third layer(convolutional layer)
        nn.Conv2d(in_channels=256, out_channels=384, stride=1, padding=1),
        nn.ReLU(inplace=True),
        # fourth layer(convolutional layer)
        nn.Conv2d(in_channels=384, out_channels=384, stride=1, padding=1),
        nn.ReLU(inplace=True),
        # fifth layer(convolutional layer)
        nn.Conv2d(in_channels=384, out_channels=256, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernal_size=3, stride=2)
    )

    # dense layers(include sixth, seventh, eighth FC layers)
    self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
