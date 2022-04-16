#!/usr/bin/python3
# coding=utf-8

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


########################### Data Augmentation ###########################
class Normalize_imd(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image, mask, depth):
        image = (image - self.mean) / self.std
        mask /= 255
        depth /= 255
        return image, mask, depth


class RandomCrop_imd(object):
    def __call__(self, image, mask, depth):
        H, W, _ = image.shape
        randw = np.random.randint(W / 8)
        randh = np.random.randint(H / 8)
        offseth = 0 if randh == 0 else np.random.randint(randh)
        offsetw = 0 if randw == 0 else np.random.randint(randw)
        p0, p1, p2, p3 = offseth, H + offseth - randh, offsetw, W + offsetw - randw
        return image[p0:p1, p2:p3, :], mask[p0:p1, p2:p3], depth[p0:p1, p2:p3]


class RandomFlip_imd(object):
    def __call__(self, image, mask, depth):
        if np.random.randint(2) == 0:
            return image[:, ::-1, :], mask[:, ::-1], depth[:, ::-1]
        else:
            return image, mask, depth


########################### Data Augmentation ###########################
class Resize_imd(object):
    def __init__(self, H, W):
        self.H = H
        self.W = W

    def __call__(self, image, mask, depth):
        image = cv2.resize(image, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        depth = cv2.resize(depth, dsize=(self.W, self.H), interpolation=cv2.INTER_LINEAR)
        return image, mask, depth


class ToTensor_imd(object):
    def __call__(self, image, mask, depth):
        image = torch.from_numpy(image)
        image = image.permute(2, 0, 1)
        mask = torch.from_numpy(mask)
        return image, mask, depth


########################### Config File ###########################
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.mean = np.array([[[124.55, 118.90, 102.94]]])
        self.std = np.array([[[56.77, 55.97, 57.50]]])
        print('\nParameters...')
        for k, v in self.kwargs.items():
            print('%-10s: %s' % (k, v))

    def __getattr__(self, name):
        if name in self.kwargs:
            return self.kwargs[name]
        else:
            return None


########################### Dataset Class ###########################
class Data(Dataset):
    def __init__(self, cfg):
        self.cfg = cfg
        self.normalize_imd = Normalize_imd(mean=cfg.mean, std=cfg.std)
        self.randomcrop_imd = RandomCrop_imd()
        self.randomflip_imd = RandomFlip_imd()

        self.resize_imd = Resize_imd(352, 352)
        self.totensor_imd = ToTensor_imd()
        with open(cfg.datapath + '/' + cfg.mode + '.txt', 'r') as lines:
            self.samples = []
            for line in lines:
                self.samples.append(line.strip())

    def __getitem__(self, idx):
        name = self.samples[idx]

        image = cv2.imread(self.cfg.datapath + '/Image/' + name + '.jpg')[:, :, ::-1].astype(np.float32)
        mask = cv2.imread(self.cfg.datapath + '/Mask/' + name + '.png', 0).astype(np.float32)
        depth = cv2.imread(self.cfg.datapath + '/Depth/' + name + '.png', 0).astype(np.float32)
        shape = mask.shape

        # print('mask--', shape)

        # print('--image--', image.shape)
        # print('--depth--', depth.shape)

        if self.cfg.mode == 'train':
            image, mask, depth = self.normalize_imd(image, mask, depth)  # for s # todo
            image, mask, depth = self.randomcrop_imd(image, mask, depth)
            image, mask, depth = self.randomflip_imd(image, mask, depth)

            return image, mask, depth
        else:
            image, mask, depth = self.normalize_imd(image, mask, depth)  # for s
            image, mask, depth = self.resize_imd(image, mask, depth)
            image, mask, depth = self.totensor_imd(image, mask, depth)

            return image, mask, depth, shape, name  # todo dont need edge

    def collate(self, batch):
        size = [288, 320, 352][np.random.randint(0, 3)]
        image, mask, depth = [list(item) for item in zip(*batch)]  # todo zzz

        for i in range(len(batch)):
            image[i] = cv2.resize(image[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            mask[i] = cv2.resize(mask[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)
            depth[i] = cv2.resize(depth[i], dsize=(size, size), interpolation=cv2.INTER_LINEAR)  # todo zzz
            # print('--image--', image[0].shape)
            # print('--depth--', depth[0].shape)

        image = torch.from_numpy(np.stack(image, axis=0)).permute(0, 3, 1, 2)
        mask = torch.from_numpy(np.stack(mask, axis=0)).unsqueeze(1)
        depth = torch.from_numpy(np.stack(depth, axis=0)).unsqueeze(1)  # todo zzz

        return image, mask, depth  # todo zzz

    def __len__(self):
        return len(self.samples)



