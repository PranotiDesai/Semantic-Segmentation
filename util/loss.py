import torch
import torch.nn as nn
from util.gradient import Gradient


class SemanticLoss(nn.Module):
    def __init__(self, n_class, class_weights=None, ignore_index=255, cuda_flag=False):
        """
        :param class_weights: class weights to apply to the loss
        :param ignore_index: index to ignore while computing loss
        """
        super().__init__()
        self.n_class = n_class
        self.ce = nn.CrossEntropyLoss(weight=class_weights, ignore_index=ignore_index, reduction="mean")
        self.grad = Gradient(cuda_flag)

        if cuda_flag:
            self.device = "cuda"
        else:
            self.device = "cpu"
        # softargmax beta value
        self.beta = 12
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, pred, target):
        b, c, h, w = pred.shape
        ce_loss = self.ce(pred, target)
        #x = self.beta * pred
        #x_max = torch.max(x, dim=1, keepdim=True)[0]
        #ex = torch.exp(x - x_max)
        #sum_ex = torch.sum(ex, dim=1, keepdim=True)
        #idx = torch.arange(pred.shape[1], device=self.device).view(1, -1, 1, 1).repeat(b, 1, h, w).float()
        #softargmax = torch.sum((ex*idx) / (sum_ex+1e-10), dim=1)

        idx = torch.arange(pred.shape[1], device=self.device).view(1, -1, 1, 1).repeat(b, 1, h, w).float()
        softargmax = torch.sum(self.softmax(self.beta*pred)*idx, dim=1)
        semantic_map_grad = self.grad(softargmax/self.n_class)
        smoothness_loss = torch.mean(semantic_map_grad)
        return ce_loss, smoothness_loss
