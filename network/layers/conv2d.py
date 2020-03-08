import torch
import torch.nn as nn


class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, dilation=1, bias=False):
        """
        :param in_ch: Input Channel
        :param out_ch: Output Channel
        :param ksize: kernel size
        :param stride: stride value
        :param bias: A boolean flag to denote if bias term needs to be include
        """
        super(Conv2D, self).__init__()
        padding = ksize // 2
        self.conv = nn.Conv2d(in_ch, out_ch, ksize, stride, padding=padding*dilation, dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x