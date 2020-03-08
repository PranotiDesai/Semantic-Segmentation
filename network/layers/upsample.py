import torch
import torch.nn.functional as F
import torch.nn as nn


class Upsample(nn.Module):
    def __init__(self, in_ch, out_ch):
        """
        :param in_ch: Input Channel
        :param out_ch: Output Channel
        """
        super(Upsample, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, 1)

    def forward(self, x, size):
        if x.shape[2] != size[0] or x.shape[3] != size[1]:
            x = F.interpolate(x, size=size, mode='bilinear', align_corners=True)
        x = self.conv(x)
        return x