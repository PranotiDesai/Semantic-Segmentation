import torch.nn as nn
from network.xception.entry_block import EntryBlock
from network.xception.middle_block import MiddleBlock
from network.xception.exit_block import ExitBlock


class XceptionNet(nn.Module):
    def __init__(self, output_stride=16):
        super(XceptionNet, self).__init__()
        entry_stride = 2
        middle_dilation = 1
        exit_dilation = (1, 2)
        if output_stride == 8:
            entry_stride = 1
            middle_dilation = 2
            exit_dilation = (2, 4)

        self.entry_block = EntryBlock(stride=entry_stride)
        self.middle_block = MiddleBlock(dilation=middle_dilation)
        self.exit_block = ExitBlock(dilation=exit_dilation)

    def forward(self, x):
        x, image_features = self.entry_block(x)
        x = self.middle_block(x)
        x = self.exit_block(x)

        return x, image_features