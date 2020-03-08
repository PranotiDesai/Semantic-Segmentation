import torch
import torch.nn as nn


class GlobalAveragePool2D(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch: Input Channel
        :param out_ch: Output Channel
        """
        super(GlobalAveragePool2D, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x