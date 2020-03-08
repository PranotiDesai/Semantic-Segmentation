import torch
import torch.nn as nn


class Skip(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3,  stride=1, bias=False):
        """
        :param in_ch: Input Channel
        :param out_ch: Output Channel
        :param ksize: kernel size
        :param stride: stride value
        :param bias: A boolean flag to denote if bias term needs to be include
        """
        super(Skip, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1, stride, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x