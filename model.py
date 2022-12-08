from torchvision import models
from torch import nn
class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.efficientNet = models.efficientnet_v2_s()
        self.fc = nn.Linear(1000,2)
    def forward(self,x):
        return self.fc(self.efficientNet(x))