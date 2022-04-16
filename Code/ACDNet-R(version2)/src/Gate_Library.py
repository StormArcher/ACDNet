import torch
import torch.nn as nn

# from module.BaseBlocks import BasicConv2d, BasicConv2d_sigmoid
# from utils.tensor_ops import cus_sample, upsample_add
# from backbone.origin.from_origin import Backbone_ResNet50_in3, Backbone_VGG16_in3
# from module.MyModule import AIM, SIM
import torch.nn.functional as F


def weight_init(module):
    for n, m in module.named_children():
        print('initialize: '+n)
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.ReLU):
            pass
        else:
            m.initialize()


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        # if self.relu is not None:
        #    x = self.relu(x)
        return x

    def initialize(self):
        weight_init(self)


class BCWH_1111(nn.Module):
    def __init__(self, channel=64):
        super(BCWH_1111, self).__init__()
        # self.conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)  # todo depth
        # self.bn = nn.BatchNorm2d(1)

        self.convS = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convD = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.convF = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, S_, D_, F_):
        # =>for compare
        S_ = torch.sigmoid((self.convS(S_)))  # 64-> 1  (B, 1, W, H)
        D_ = torch.sigmoid((self.convD(D_)))  # 64-> 1  (B, 1, W, H)
        F_ = torch.sigmoid((self.convF(F_)))  # 64-> 1  (B, 1, W, H)

        # supress S
        interS = ((F_ * S_) * 1).sum(dim=(2, 3))  # B1HW-> B111
        unionS = ((F_ + S_) * 1).sum(dim=(2, 3))
        wiouS = 1 - (interS + 1) / (unionS - interS + 1)
        wiouS = wiouS.unsqueeze(2).unsqueeze(3)  # .expand_as(D)

        # supress D
        interD = ((F_ * D_) * 1).sum(dim=(2, 3))  # B1HW-> B111
        unionD = ((F_ + D_) * 1).sum(dim=(2, 3))
        wiouD = 1 - (interD + 1) / (unionD - interD + 1)
        wiouD = wiouD.unsqueeze(2).unsqueeze(3)  # .expand_as(D)

        return wiouS, wiouD, 1

    def initialize(self):
        weight_init(self)


class GL(nn.Module):
    def __init__(self):
        super(GL, self).__init__()
        self.BCWH2 = BCWH_1111()
        self.BCWH3 = BCWH_1111()
        self.BCWH4 = BCWH_1111()
        self.BCWH5 = BCWH_1111()

    def forward(self, out5s, out5d, out5f,
                      out4s, out4d, out4f,
                      out3s, out3d, out3f,
                      out2s, out2d, out2f):

        G2s, G2d, G2f = self.BCWH2(out2s, out2d, out2f)
        G3s, G3d, G3f = self.BCWH3(out3s, out3d, out3f)
        G4s, G4d, G4f = self.BCWH4(out4s, out4d, out4f)
        G5s, G5d, G5f = self.BCWH5(out5s, out5d, out5f)

        print('2---->')
        print('G2s=>', G2s)
        print('G2d=>', G2d)
        print('G2f=>', G2f)


        print('5---->')
        print('G5s=>', G5s)
        print('G5d=>', G5d)
        print('G5f=>', G5f)

        return G2s, G3s, G4s, G5s, \
               G2d, G3d, G4d, G5d, \
               G2f, G3f, G4f, G5f,

    def initialize(self):
        weight_init(self)


class GL_CFE(nn.Module):
    def __init__(self, channel=64):
        super(GL_CFE, self).__init__()
        self.conv1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)  # todo depth
        self.bn1 = nn.BatchNorm2d(1)
        self.conv2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)  # todo depth
        self.bn2 = nn.BatchNorm2d(1)
        self.conv3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False)  # todo depth
        self.bn3 = nn.BatchNorm2d(1)

    def forward(self, x):
        Ga = torch.sigmoid(self.bn1(self.conv1(x)))  # 64 * 2-> 1   (B, 1, W, H)
        Ga= F.avg_pool2d(Ga, (Ga.size(2), Ga.size(3)), stride=(Ga.size(2), Ga.size(3)))  # Ga (B, 1, 1, 1)
        Gb = torch.sigmoid(self.bn2(self.conv2(x)))
        Gb = F.avg_pool2d(Gb, (Gb.size(2), Gb.size(3)), stride=(Gb.size(2), Gb.size(3)))
        Gc = torch.sigmoid(self.bn3(self.conv3(x)))
        Gc = F.avg_pool2d(Gc, (Gc.size(2), Gc.size(3)), stride=(Gc.size(2), Gc.size(3)))

        return Ga, Gb, Gc

    def initialize(self):
        weight_init(self)