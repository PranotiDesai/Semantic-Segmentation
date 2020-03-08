import os
import torch
import numpy as np
from dataloader.ADEK_dataloader import ADEKDataLoader


def compute_class_weights(n_class, src_dir, cuda_flag=False):
    """
    This method computes the class weights by iterating through the dataloader
    :param n_class: no of unique classes in the dataset
    :return:
    """
    if os.path.exists("metadata/class_weights.npy"):
        return torch.from_numpy(np.load("metadata/class_weights.npy"))

    class_counts = torch.zeros(n_class).float()
    dataloader, _ = ADEKDataLoader(src_dir, batch_size=1, target_dim=None, nproc=0).get_loader()
    total_pixels = 0
    for i, data in enumerate(dataloader):
        if cuda_flag:
            # data['image'] = data['image'].cuda()
            data['annotations'] = data['annotations'].cuda()
        total_pixels += data['annotations'].nelement()
        annotations = data['annotations']
        if i % 1000 == 0:
            print(i)
        if cuda_flag:
            annotations = annotations.cuda()
        for c in range(n_class):
            class_counts[c] += torch.sum(annotations == c).float()/data['annotations'].nelement()

    class_count_ratio = class_counts.float()/len(dataloader)
    class_weights = 1/(class_count_ratio + 1e-6)
    np.save("metadata/class_weights.npy", class_weights.numpy())
    return class_weights


def load_model(model, filename):
    state_dict = torch.load(filename)
    model_state_dict = model.state_dict()
    pretrained_dict = dict()
    for k, v in state_dict.items():
        if k in model_state_dict and v.shape == model_state_dict[k].shape:
            pretrained_dict[k] = v
        else:
            pretrained_dict[k] = model_state_dict[k]
    model_state_dict.update(pretrained_dict)
    model.load_state_dict(pretrained_dict)
    return model