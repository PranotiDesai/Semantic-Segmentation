import torch.nn as nn
from network.layers.conv2d import Conv2D
from network.xception.xception_block import ExceptionBlock


class EntryBlock(nn.Module):
    def __init__(self, stride):
        """
        :param stride: Stride to be used for entry block
        """
        super(EntryBlock, self).__init__()
        self.conv1 = Conv2D(3, 32, 3, 2)
        self.conv2 = Conv2D(32, 64, 3, 1)

        self.block1 = ExceptionBlock(64, 128, 3, 2, repeat=2, relu_begin=False, grow_first=False)
        self.block2 = ExceptionBlock(128, 256, 3, 2, repeat=2, relu_begin=False)
        self.block3 = ExceptionBlock(256, 728, 3, repeat=2, is_last=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.block1(x)
        image_features = x
        x = self.block2(x)
        x = self.block3(x)
        return x, image_features
