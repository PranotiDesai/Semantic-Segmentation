import torch
import torch.nn as nn


class DepthWiseSeparableConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3, stride=1, dilation=1, bias=True):
        """
        :param in_ch: Input Channel
        :param out_ch: Output Channel
        :param ksize: kernel size
        :param stride: stride value
        :param bias: A boolean flag to denote if bias term needs to be include
        """
        super(DepthWiseSeparableConvolution, self).__init__()
        padding = ksize // 2
        padding *= dilation
        self.conv = nn.Conv2d(in_ch, in_ch, ksize, stride, padding=padding, groups=in_ch, dilation=dilation, bias=bias)
        self.pointwise_conv = nn.Conv2d(in_ch, out_ch, 1, 1, padding=0, dilation=1, groups=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv(x)
        x = self.pointwise_conv(x)
        x = self.bn(x)

        return x