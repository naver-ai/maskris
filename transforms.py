import numpy as np

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class ColorJitter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.color_jitter = T.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation, hue=hue)

    def __call__(self, image, target):
        image = self.color_jitter(image)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target):
        if np.random.rand() < self.prob:
            image = F.hflip(image)
            if isinstance(target, list):
                target_new = []
                for _target in target:
                    target_new.append(F.hflip(_target))
                target = target_new
            else:
                target = F.hflip(target)
        return image, target


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, image, target):
        i, j, h, w = T.RandomResizedCrop.get_params(image, scale=self.scale, ratio=self.ratio)
        image = F.resized_crop(image, i, j, h, w, self.size)
        if isinstance(target, list):
            target_new = []
            for _target in target:
                target_new.append(F.resized_crop(_target, i, j, h, w, self.size, interpolation=F.InterpolationMode.NEAREST))
            target = target_new
        else:
            target = F.resized_crop(target, i, j, h, w, self.size, interpolation=F.InterpolationMode.NEAREST)
        return image, target


class Resize(object):
    def __init__(self, h, w, eval_mode=False):
        self.h = h
        self.w = w
        self.eval_mode = eval_mode

    def __call__(self, image, target):
        image = F.resize(image, (self.h, self.w))
        # If size is a sequence like (h, w), the output size will be matched to this.
        # If size is an int, the smaller edge of the image will be matched to this number maintaining the aspect ratio
        if not self.eval_mode:
            if isinstance(target, list):
                target_new = []
                for _target in target:
                    target_new.append(F.resize(_target, (self.h, self.w), interpolation=F.InterpolationMode.NEAREST))
                target = target_new
            else:
                target = F.resize(target, (self.h, self.w), interpolation=F.InterpolationMode.NEAREST)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        if isinstance(target, list):
            target_new = []
            for _target in target:
                target_new.append(torch.as_tensor(np.asarray(_target).copy(), dtype=torch.int64))
            target = target_new
        else:
            target = torch.as_tensor(np.asarray(target).copy(), dtype=torch.int64)
        return image, target

class Normalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, target):
        image = F.normalize(image, mean=self.mean, std=self.std)
        return image, target

