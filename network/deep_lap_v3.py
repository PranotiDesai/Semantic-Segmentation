import torch
import torch.nn as nn
from network.xception.xception_net import XceptionNet
from network.aspp.aspp import ASPP
from network.decoder.decoder import Decoder
from network.layers.upsample import Upsample


class DeepLabV3Plus(nn.Module):
    def __init__(self, n_class, output_stride=16):
        super(DeepLabV3Plus, self).__init__()

        self.xception = XceptionNet(output_stride)
        self.aspp = ASPP(output_stride)
        self.decoder = Decoder(n_class )
        self.upsample = Upsample(n_class, n_class)
        self.init_weight()
        self.get_parameters_count()

    def forward(self, x):
        b, c, h, w = x.shape
        x, image_features = self.xception(x)
        multi_scale_features = self.aspp(x)
        x = self.decoder(multi_scale_features, image_features)
        x = self.upsample(x, (h, w))

        return x

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def get_parameters_count(self):
        count = 0
        for parameter in self.parameters():
            count += parameter.numel()

        print("Total Parameters: %d" % count)