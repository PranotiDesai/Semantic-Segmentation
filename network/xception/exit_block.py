import torch.nn as nn
from network.xception.xception_block import ExceptionBlock
from network.layers.depthwise_seperable_convolution import DepthWiseSeparableConvolution as DSConv


class ExitBlock(nn.Module):
    def __init__(self, dilation):
        """
        :param dilation: Dilation to be used
        """
        super(ExitBlock, self).__init__()
        self.relu = nn.ReLU()
        self.block1 = ExceptionBlock(728, 1024, 3, stride=1, repeat=2, dilation=dilation)
        self.ds_conv1 = DSConv(1024, 1536, 3, stride=1, dilation=dilation)
        self.ds_conv2 = DSConv(1536, 1536, 3, stride=1, dilation=dilation)
        self.ds_conv3 = DSConv(1536, 2048, 3, stride=1, dilation=dilation)

    def forward(self, x):
        x = self.block1(x)
        x = self.ds_conv1(x)
        x = self.ds_conv2(x)
        x = self.ds_conv3(x)

        return x