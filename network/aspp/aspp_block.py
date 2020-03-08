import torch.nn as nn
from network.layers.conv2d import Conv2D


class ASPP(nn.Module):
    def __init__(self, in_ch, out_ch, ksize, stride, dilation, bias=False):
        """
        :param in_ch: Input Channel
        :param out_ch: output Channel
        :param ksize: kernel size
        :param stride: stride
        :param dilation: dilation
        :param bias: a boolean flag to denote if need to include bias term
        """
        super(ASPP, self).__init__()
        self.conv = Conv2D(in_ch, out_ch, ksize, stride, dilation, padding=dilation, bias=bias)

    def forward(self, x):
        x = self.conv(x)
        x = self.middle_block(x)
        x = self.exit_block(x)

        return x