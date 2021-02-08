import torch.nn as nn
import torch.nn.functional as F 
from typing import Any

class AlexNet(nn.Module):

  def __init__(self, num_classes: int = 1000) -> None:
    super(AlexNet, self).__init__()
    self.features = nn.Sequential(
        #first layer(convolutional layer)
        nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #second layer(convolutional layer)
        nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
        nn.ReLU(inplace=True),
        nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
        nn.MaxPool2d(kernel_size=3, stride=2),
        #third layer(convolutional layer)
        nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        #fourth layer(convolutional layer)
        nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        #fifth layer(convolutional layer)
        nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2)
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
    
    self.init_bias()
    
  # initialize weights and bias
  def init_bias(self):
    for layer in self.features:
      if isinstance(layer, nn.Conv2d):
        nn.init.normal_(layer.weight, mean=0, std=0.01)
        nn.init.constant_(layer.bias, 0)
     
    nn.init.constant_(self.features[4].bias, 1)
    nn.init.constant_(self.features[10].bias, 1)
    nn.init.constant_(self.features[12].bias, 1)
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
      x = self.features(x)
      x = torch.flatten(x, 1)
      x = self.classifier(x)  
      return x

net = AlexNet()
