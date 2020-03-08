import torch
import torch.nn as nn


class Gradient(nn.Module):
    def __init__(self, cuda_flag=False):
        super(Gradient, self).__init__()
        self.Kx = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]]).float()
        self.Ky = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]]).float()
        self.grad_X = nn.Conv2d(1, 1, 3, 1, padding=1)
        self.grad_Y = nn.Conv2d(1, 1, 3, 1, padding=1)
        self.grad_X.weight.data.copy_(self.Kx)
        self.grad_Y.weight.data.copy_(self.Ky)

        if cuda_flag:
            self.grad_X = self.grad_X.cuda()
            self.grad_Y = self.grad_Y.cuda()

    def forward(self, x):
        x = x.unsqueeze(1)
        grad_x = self.grad_X(x)  # normalize by the class count
        grad_y = self.grad_Y(x)
        grad = torch.pow(torch.pow(grad_x, 2) + torch.pow(grad_y, 2), 0.5)
        return grad