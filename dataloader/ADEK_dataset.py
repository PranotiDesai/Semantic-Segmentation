import os
import cv2
import torch
import numpy as np
from imgaug import augmenters as iaa
from torch.utils.data.dataset import Dataset
from torchvision.transforms import transforms
from dataloader.transformations import ToTensor


class ADEKDataset(Dataset):
    def __init__(self, src_dir, is_train=True, target_dim=(256, 256)):
        self.target_dim = target_dim
        self.object_info = dict()
        self.class_weights = []
        self.image_filenames = []
        self.annotation_filenames = []
        self.is_train = is_train
        if is_train:
            images_dir = src_dir+"images/training/"
            annotations_dir = src_dir+"annotations/training/"
            if target_dim is None: # used in case of class weight computation
                self.transform = transforms.Compose([
                    ToTensor()
                ])
            else:
                self.size_aug = iaa.Sequential([
                    iaa.PadToFixedSize(target_dim[0], target_dim[1]),
                    iaa.CropToFixedSize(target_dim[0], target_dim[1])
                ])
                self.property_aug = iaa.Sequential([
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01 * 255), per_channel=0.3),
                    iaa.AddToHueAndSaturation((-20, 20))
                ])
        else:
            images_dir = src_dir + "images/validation/"
            annotations_dir = src_dir + "annotations/validation/"
            self.size_aug = iaa.Sequential([
                iaa.PadToFixedSize(target_dim[0], target_dim[1]),
                iaa.CropToFixedSize(target_dim[0], target_dim[1])
            ])
        files = os.listdir(images_dir)
        for file in files:
            filename = file.split(".")[0]
            self.image_filenames.append(images_dir+filename+".jpg")
            self.annotation_filenames.append(annotations_dir+filename+".png")

        print(len(self.image_filenames))

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, item_index):
        image = cv2.imread(self.image_filenames[item_index])[:, :, ::-1].copy()
        # Since the annotations are gray scale images hence load them as grayscale
        annotation = cv2.imread(self.annotation_filenames[item_index], cv2.IMREAD_UNCHANGED)
        annotation = np.expand_dims(np.expand_dims(annotation, 2), 0)

        data = dict()
        if self.size_aug:
            data['images'], data['annotations'] = self.size_aug(image=image, segmentation_maps=annotation)
            if self.is_train:
                data['images'], data['annotations'] = self.property_aug(image=data['images'],
                                                                        segmentation_maps=data['annotations'])

        data['images'] = data['images']/255
        data['annotations'] = data['annotations'].squeeze()
        data['images'] = torch.from_numpy(np.transpose(data['images'], [2, 0, 1]).copy()).float()
        data['annotations'] = torch.from_numpy(data['annotations'].copy()).long()
        return data
