import torch
import torch.nn as nn
from util.adek_colormap import ADEKColormap


class SegmentationVisualizer(nn.Module):
    def __init__(self, writer, n_class, cuda_flag=False):
        """
        :param writer: Tensorboard Summary Writer
        :param n_class: Number of classes
        :param cuda_flag: cuda flag
        """
        super(SegmentationVisualizer, self).__init__()
        self.writer = writer
        self.colormap = ADEKColormap(n_class, cuda_flag)

    def forward(self, images, target, pred, step):
        """
        :param images: Input Images
        :param target: Ground Truth Target
        :param pred: Predictions
        :param step: step for tensor board
        :return:
        """
        # concat the target and pred as batch so the embedding lookup can be performed in one go
        size = min(6, pred.shape[0])
        images = images[0:size]
        pred = pred[0:size]
        target = target[0:size]
        pred_target_batch = torch.cat((pred.long(), target.long()), dim=0)
        colormap = self.colormap(pred_target_batch)
        pred = colormap[:size]
        target = colormap[size:]

        # concat the images as
        self.writer.add_images("Images", images, step)
        self.writer.add_images("Prediction", pred, step)
        self.writer.add_images("Target", target, step)

