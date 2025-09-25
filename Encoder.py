import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, ENCODER_DIM=512):
        super(Encoder, self).__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),
            nn.Conv2d(256, ENCODER_DIM, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(ENCODER_DIM),
            nn.ReLU(),
        )

    def forward(self, images):
        return self.cnn(images)
