import torch.nn as nn
from network.xception.xception_block import ExceptionBlock


class MiddleBlock(nn.Module):
    def __init__(self, dilation):
        """
        :param dilation: Dilation to be used
        """
        super(MiddleBlock, self).__init__()
        self.block1 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block2 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block3 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block4 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block5 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block6 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block7 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block8 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block9 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block10 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block11 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block12 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block13 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block14 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block15 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)
        self.block16 = ExceptionBlock(728, 728, 3, 1, repeat=3, dilation=dilation)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.block13(x)
        x = self.block14(x)
        x = self.block15(x)
        x = self.block16(x)

        return x