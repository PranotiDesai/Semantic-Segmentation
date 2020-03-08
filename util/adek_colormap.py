import torch
import torch.nn as nn
import numpy as np


class ADEKColormap(nn.Module):
    def __init__(self, n_class, cuda_flag):
        super(ADEKColormap, self).__init__()

        cmap = np.zeros((n_class, 3))

        # generate colormap based on constant color difference among classes
        for i in range(150):
            cmap[i] = i * np.array([7, 33, 57])
        cmap = torch.from_numpy(cmap).type(torch.uint8)
        self.colormap = nn.Embedding(cmap.shape[0], 3)
        self.colormap.weight.data.copy_(cmap)

        if cuda_flag:
            self.colormap = self.colormap.cuda()

    def forward(self, x):
        with torch.no_grad():
            return self.colormap(x.unsqueeze(1)).squeeze(1).permute(0, 3, 1, 2).float()/255