import torch.nn as nn
import torch.nn.functional as functional
import torch.nn.init as init
from torch.nn.modules.linear import Linear

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 28, 2),
            nn.ReLU(),
            nn.Conv2d(28, 28, 2),
            nn.BatchNorm2d(28),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(28, 56, 2),
            nn.ReLU(),
            nn.Conv2d(56, 56, 3),
            nn.BatchNorm2d(56),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.5)
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(56, 112, 3),
            nn.ReLU(),
            nn.Conv2d(112, 112, 3),
            nn.BatchNorm2d(112),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5)
        )
        
        self.fc = nn.Linear(1008, 36)

        
    def forward(self, x):   

        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        
        return out