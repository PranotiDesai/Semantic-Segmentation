import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from network.layers.conv2d import Conv2D
from network.layers.average_pool import GlobalAveragePool2D


class ASPP(nn.Module):
    def __init__(self, output_stride=16):
        dilations = np.array([1, 6, 12, 18])
        if output_stride == 8:
            dilations[1:] *= 2
        super(ASPP, self).__init__()
        self.aspp1 = Conv2D(2048, 256, 1, 1, dilation=dilations[0])
        self.aspp2 = Conv2D(2048, 256, 3, 1, dilation=dilations[1])
        self.aspp3 = Conv2D(2048, 256, 3, 1, dilation=dilations[2])
        self.aspp4 = Conv2D(2048, 256, 3, 1, dilation=dilations[3])
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.avg_pool_conv = Conv2D(2048, 256, 1)
        self.conv = Conv2D(256*5, 256, 1)
        # self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        aspp1 = self.aspp1(x)
        aspp2 = self.aspp2(x)
        aspp3 = self.aspp3(x)
        aspp4 = self.aspp4(x)
        avg_pool = self.avg_pool_conv(self.avg_pool(x))
        avg_pool = F.interpolate(avg_pool, size=aspp1.shape[2:])
        x = torch.cat((aspp1, aspp2, aspp3, aspp4, avg_pool), dim=1)
        x = self.conv(x)
        # x = self.dropout(x)

        return x