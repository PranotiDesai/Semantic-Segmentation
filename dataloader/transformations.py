import cv2
import torch
import numpy as np
from imgaug import augmenters as iaa


class RandomCrop(object):
    """
    This class is responsible for cropping the image and removing the annotations which does not fall in the cropped
    region.
    """
    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, data):
        image = data['image']
        annotations = data['annotations']
        h, w = image.shape[0:2]
        if h < self.crop_size[0] or w < self.crop_size[1]:
            hfactor = self.crop_size[0]/h
            wfactor = self.crop_size[1]/w
            if hfactor > wfactor:
                factor = hfactor
            else:
                factor = wfactor
            image = cv2.resize(image, None, fx=factor,  fy=factor, interpolation=cv2.INTER_LINEAR)
            annotations = cv2.resize(annotations, None, fx=factor,  fy=factor, interpolation=cv2.INTER_NEAREST)

            h, w = image.shape[0:2]
        try:
            diff = w - self.crop_size[1]
            if diff == 0:
                x1 = 0
            else:
                x1 = np.random.randint(0, diff)

            diff = h - self.crop_size[0]
            if diff == 0:
                y1 = 0
            else:
                y1 = np.random.randint(0, diff)
        except:
            print("Error")
        x2 = x1 + self.crop_size[1]
        image = image[:, x1:x2]
        annotations = annotations[:, x1:x2]
        y2 = y1 + self.crop_size[0]
        image = image[y1:y2, :]
        annotations = annotations[y1:y2, :]

        data['image'] = image
        data['annotations'] = annotations
        return data


class Resize:
    def __init__(self, size):
        self.size = size

    def __call__(self, data):
        data['image'] = cv2.resize(data['image'], self.size, cv2.INTER_LINEAR)
        data['annotations'] = cv2.resize(data['annotations'], self.size, cv2.INTER_NEAREST)
        return data


class Normalize(object):
    def __call__(self, data):
        data['image'] = data['image']/255
        return data


class PropertyAugmentation(object):
    def __init__(self):
        self.propertyaug = iaa.Sequential([
                    iaa.Multiply((0.8, 1.2), per_channel=0.2),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.3),
                    iaa.AddToHueAndSaturation((-20, 20))
        ])

    def __call__(self, data):
        image = data['image']
        annotations = data['annotations']
        if np.random.random() < 0.5:
            image = np.fliplr(image)
            annotations = np.fliplr(annotations)
        image = self.propertyaug(image=image)
        data['image'] = image
        data['annotations'] = annotations
        return data


class ToTensor(object):
    def __call__(self, data):
        data['image'] = torch.from_numpy(np.transpose(data['image'], [2, 0, 1]).copy()).float()
        data['annotations'] = torch.from_numpy(data['annotations'].copy()).long()
        return data

