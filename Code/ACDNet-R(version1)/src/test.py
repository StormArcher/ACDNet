# !/usr/bin/python3
# coding=utf-8

import os
import sys

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.ion()

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import dataset
from net import ACDNet
 

# <datasets Documentation>
# 8 RGB-D datasets : NJUD(NJU2K)(1985 pictures), NLPR(1000 pic), RGBD135(135), SIP(929), SSD(80), STEREO(1000), LFSD, DUTRGBD(1200)
# 1st: we choose 1485 images from NJUD and 700 images from NLPR as the training set (NJUDNLPR-TR), and the remaining data of these datasets except DUTRGBD as the testing set.
# 2nd: For DUTRGBD, the proposed model is trained on 800 images and tested on the remaining 400 images.

# <model name>
# EP = '30-DUTRGBD' #   1st model name: model-30-DUTRGBD is trained by training set (DUTRGBD)
EP = '30-NJUDNLPR-TR' # 2nd model name: model-30-v3 is trained by training set (NJUDNLPR-TR)

# <algorithom name>
version = '-ACDNet-R' # algorithom name


class Test(object):
    def __init__(self, Dataset, Network, path):
        ## dataset
        self.cfg = Dataset.Config(datapath=path, snapshot='./out/model-'+EP, mode='test')  # model root: ./out/model-30-NJUDNLPR-TR
        self.data = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net = Network(self.cfg)
        self.net.train(False) 
        self.net.cuda()


    def save(self):
        with torch.no_grad():
            print('<=Show Dateset=> ', self.cfg.datapath.split('/')[-1])  # to show the dataset 

            for image, mask, depth, shape, name in self.loader: 
                image = image.cuda().float()
                depth = depth.cuda().float().unsqueeze(1)
                out2, GS, GS3, GS4, GD, GD3, GD4 = self.net(image, depth, shape)
                out2 = (torch.sigmoid(out2[0, 0]) * 255).cpu().numpy()
                head = '../eval/maps/ACDNet/' + self.cfg.datapath.split('/')[-1] + version + '-'+EP
                # root is ../eval/maps/ACDNet/SSD-ACDNet-R-30-NJUDNLPR-TR
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(out2))  # imwrite


if __name__ == '__main__': 
    for path in [
'/home/aaa/DL/data/data-RGBD/test/SSD', 
'/home/aaa/DL/data/data-RGBD/test/STEREO',
'/home/aaa/DL/data/data-RGBD/test/LFSD',
'/home/aaa/DL/data/data-RGBD/test/NJU2K-Test', 
'/home/aaa/DL/data/data-RGBD/test/NLPR-Test',
'/home/aaa/DL/data/data-RGBD/test/RGBD135', 
'/home/aaa/DL/data/data-RGBD/test/SIP' ,
#'/home/aaa/DL/data/data-RGBD/test/DUT-RGBD-Test'
              ]:
        t = Test(dataset, ACDNet, path)
        t.save()
