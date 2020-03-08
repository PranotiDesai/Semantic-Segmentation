import torch
import torch.nn.functional as F
import torch.nn as nn
from network.layers.conv2d import Conv2D


class Decoder(nn.Module):
    def __init__(self, n_class=10):
        super(Decoder, self).__init__()
        self.conv1 = Conv2D(128, 48, 1, 1)
        self.conv2 = Conv2D(256, 256, 1, 1)
        self.conv3 = Conv2D(304, 256, 3, 1)
        self.conv4 = nn.Conv2d(256, n_class, 1, 1)

    def forward(self, x, image_features):
        x = F.interpolate(x, image_features.shape[2:], mode="bilinear", align_corners=True)
        image_features = self.conv1(image_features)
        x = self.conv2(x)
        x = torch.cat((x, image_features), dim=1)
        x = self.conv3(x)
        x = self.conv4(x)

        return x