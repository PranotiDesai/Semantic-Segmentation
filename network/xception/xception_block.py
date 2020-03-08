import torch.nn as nn
from network.layers.skip import Skip
from network.layers.depthwise_seperable_convolution import DepthWiseSeparableConvolution as DSConv


class ExceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, ksize=3,  stride=1, dilation=1, repeat=2, bias=True, relu_begin=True, grow_first=True,
                 skip_connection=True, is_last=False):
        """
        :param in_ch: Input Channel
        :param out_ch: output Channel
        :param ksize: kernel size
        :param stride: stride
        :param dilation: dilation
        :param bias: a boolean flag to denote if need to include bias term
        :param skip_connection: A flag to represent if skip connection is required
        """
        super(ExceptionBlock, self).__init__()
        self.relu_begin = relu_begin
        self.repeat = repeat
        if skip_connection:
            self.skip_connection = True
            self.skip = None
            if stride != 1 or in_ch != out_ch:
                self.skip = Skip(in_ch, out_ch, 1, stride, bias=bias)

        if grow_first:
            out = out_ch
        else:
            out = in_ch
        self.ds_conv1 = DSConv(in_ch, out, ksize, stride=1, dilation=dilation, bias=bias)
        if self.repeat == 3:
            self.ds_conv2 = DSConv(out, out, ksize, stride=1, dilation=dilation, bias=bias)
            self.ds_conv3 = DSConv(out, out_ch, ksize, stride=1, dilation=dilation, bias=bias)
        else:
            self.ds_conv2 = DSConv(out, out_ch, ksize, stride=1, dilation=dilation, bias=bias)

        self.stride_ds_conv = None
        if stride >= 1:
            self.stride_ds_conv = DSConv(out_ch, out_ch, ksize, stride=stride, dilation=dilation, bias=bias)
        if stride == 1 and is_last:
            self.stride_ds_conv = DSConv(out_ch, out_ch, ksize, stride=1, dilation=dilation, bias=bias)
        self.relu = nn.ReLU()

    def forward(self, x):
        x_init = x
        if self.relu_begin:
            x = self.relu(x)
        x = self.ds_conv1(x)
        x = self.relu(x)
        x = self.ds_conv2(x)
        x = self.relu(x)
        if self.repeat > 2:
            x = self.ds_conv3(x)
            x = self.relu(x)

        if self.stride_ds_conv:
            x = self.stride_ds_conv(x)
        if self.skip_connection:
            if self.skip is not None:
                x_init = self.skip(x_init)
            x += x_init
        return x