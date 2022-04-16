#!/usr/bin/python3
#coding=utf-8
# change

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from cfe_v7 import CFE, AT  #
from Gate_Library import GL
from gate import Gate


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
        elif isinstance(m, nn.ModuleList):
            pass
        else:
            m.initialize()


# ==========>Res<==========
class Bottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, dilation=1):
        super(Bottleneck, self).__init__()
        self.conv1      = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1        = nn.BatchNorm2d(planes)
        self.conv2      = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=(3*dilation-1)//2, bias=False, dilation=dilation)
        self.bn2        = nn.BatchNorm2d(planes)
        self.conv3      = nn.Conv2d(planes, planes*4, kernel_size=1, bias=False)
        self.bn3        = nn.BatchNorm2d(planes*4)
        self.downsample = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            x = self.downsample(x)
        return F.relu(out+x, inplace=True)


# =>Encoder_R
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1_ = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out1 = F.max_pool2d(out1_, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1_, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/home/aaa/DL/pre-train/res/resnet50-19c8e357a.pth'), strict=False)


# => Encoder-D
class ResNet_D(nn.Module):
    def __init__(self):
        super(ResNet_D, self).__init__()
        self.inplanes = 64
        # self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # todo depth
        # self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x = self.conv1(x)
        # out1_ = F.relu(self.bn1(x), inplace=True)
        out1_ = x
        out1 = F.max_pool2d(out1_, kernel_size=3, stride=2, padding=1)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return out1_, out2, out3, out4, out5

    def initialize(self):
        self.load_state_dict(torch.load('/home/aaa/DL/pre-train/res/resnet50-19c8e357b.pth'), strict=False)


# Encoder-F(DCS)
class ResNet_3(nn.Module):
    def __init__(self):
        super(ResNet_3, self).__init__()
        self.inplanes = 64
        # self.conv1    = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)  # todo depth
        # self.bn1      = nn.BatchNorm2d(64)
        self.layer1   = self.make_layer( 64, 3, stride=1, dilation=1)
        self.layer2   = self.make_layer(128, 4, stride=2, dilation=1)
        self.layer3   = self.make_layer(256, 6, stride=2, dilation=1)
        self.layer4   = self.make_layer(512, 3, stride=2, dilation=1)

        self.squeeze3 = nn.Sequential(nn.Conv2d(512, 256, 1), nn.BatchNorm2d(256))
        self.squeeze4 = nn.Sequential(nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512))

        self.squeeze5t64 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4t64 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3t64 = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

    def make_layer(self, planes, blocks, stride, dilation):
        downsample    = nn.Sequential(nn.Conv2d(self.inplanes, planes*4, kernel_size=1, stride=stride, bias=False), nn.BatchNorm2d(planes*4))
        layers        = [Bottleneck(self.inplanes, planes, stride, downsample, dilation=dilation)]
        self.inplanes = planes*4
        for _ in range(1, blocks):
            layers.append(Bottleneck(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, outS, outD, out2sf, out2df, out3sf, out3df, out4sf, out4df, GS, GD, GS3, GD3, GS4, GD4):
        # x = self.conv1(x)
        # out1_ = F.relu(self.bn1(x), inplace=True)
        # out1_ = x
        # out1 = F.max_pool2d(out1_, kernel_size=3, stride=2, padding=1)
        out2 = torch.cat((out2sf, out2df, outS *GS, outD *GD), 1)  # 64+64 + 64+64
        # out2 = self.layer1(out1) # without layer1
        out3 = self.layer2(out2)
        out3 = self.squeeze3(out3)
        out3 = torch.sigmoid(out3)
        # out3 : 512->256
        out4 = self.layer3(torch.cat((out3, out3sf *GS3, out3df *GD3),1))  # 256 + 128+128
        out4 = self.squeeze4(out4)
        out4 = torch.sigmoid(out4)
        # out4 : 1024->512
        out5 = self.layer4(torch.cat((out4, out4sf *GS4, out4df *GD4),1))  # 512 + 256+256

        return out2sf, out2df, out3, out4, out5  # 64,64, 256, 512, 2048

    def initialize(self):
        self.load_state_dict(torch.load('/home/aaa/DL/pre-train/res/resnet50-19c8e357c.pth'), strict=False)


# ==========>DenseLayer<===========
class BasicConvRelu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConvRelu, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

    def initialize(self):
        weight_init(self)


# =>not used
class DenseLayer(nn.Module):
    def __init__(self, in_C, out_C, down_factor=2, k=4): 
        super(DenseLayer, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConvRelu(mid_C * i, mid_C, 3, 1, 1))  #

        self.fuse = BasicConvRelu(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)  # in_C->out_C//4=mid_C   64->16
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return self.fuse(feats)

    def initialize(self):
        weight_init(self)


# =>not used
class DenseLayer_SIG(nn.Module):
    def __init__(self, in_C, out_C, down_factor=2, k=4): 
        super(DenseLayer_SIG, self).__init__()
        self.k = k
        self.down_factor = down_factor
        mid_C = out_C // self.down_factor

        self.down = nn.Conv2d(in_C, mid_C, 1)

        self.denseblock = nn.ModuleList()
        for i in range(1, self.k + 1):
            self.denseblock.append(BasicConvRelu(mid_C * i, mid_C, 3, 1, 1))

        self.fuse = BasicConvRelu(in_C + mid_C, out_C, kernel_size=3, stride=1, padding=1, relu=False)

    def forward(self, in_feat):
        down_feats = self.down(in_feat)  # in_C->out_C//4=mid_C   64->16
        out_feats = []
        for denseblock in self.denseblock:
            feats = denseblock(torch.cat((*out_feats, down_feats), dim=1))
            out_feats.append(feats)
        feats = torch.cat((in_feat, feats), dim=1)
        return F.sigmoid(self.fuse(feats))

    def initialize(self):
        weight_init(self)


# =>one part of FPN
class FPN(nn.Module):
    def __init__(self):
        super(FPN, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, L, H):  # L low-level feature H high-level feature
        if L.size()[2:] != H.size()[2:]:
            H = F.interpolate(H, size=L.size()[2:], mode='bilinear')
        L = L + H
        L = F.relu(self.bn(self.conv(L)), inplace=True)
        return L

    def initialize(self):
        weight_init(self)


# =>one part of FPN
class FPN_sig(nn.Module):
    def __init__(self):
        super(FPN_sig, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, L, H):
        if L.size()[2:] != H.size()[2:]:
            H = F.interpolate(H, size=L.size()[2:], mode='bilinear')
        L = L + H
        L = torch.sigmoid(self.bn(self.conv(L)))
        return L

    def initialize(self):
        weight_init(self)


# =>not used
class FPN_(nn.Module):
    def __init__(self):
        super(FPN_, self).__init__()
        self.conv = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(128)

    def forward(self, L, H):
        if L.size()[2:] != H.size()[2:]:
            H = F.interpolate(H, size=L.size()[2:], mode='bilinear')
        L = L + H
        L = F.relu(self.bn(self.conv(L)), inplace=True)
        return L

    def initialize(self):
        weight_init(self)


# =>not used
class CAT(nn.Module):
    def __init__(self):
        super(CAT, self).__init__()
        self.conv = nn.Conv2d(64*4, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, out2, out3, out4, out5):
        out3 = F.interpolate(out3, size=out2.size()[2:], mode='bilinear')
        out4 = F.interpolate(out4, size=out2.size()[2:], mode='bilinear')
        out5 = F.interpolate(out5, size=out2.size()[2:], mode='bilinear')
        out = F.relu(self.bn(self.conv(torch.cat((out2, out3, out4, out5), 1))), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# =>not used
class PFAN(nn.Module):
    def __init__(self):
        super(PFAN, self).__init__()
        self.conv1 = nn.Conv2d(64*2, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64 * 2, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, out2, out3, out4, out5):
        out3 = F.interpolate(out3, size=out2.size()[2:], mode='bilinear')
        out5 = F.interpolate(out5, size=out4.size()[2:], mode='bilinear')
        outA = F.relu(self.bn1(self.conv1(torch.cat((out2, out3), 1))), inplace=True)
        outB = F.relu(self.bn2(self.conv2(torch.cat((out4, out5), 1))), inplace=True)
        outB = F.interpolate(outB, size=outA.size()[2:], mode='bilinear')
        out = F.relu(self.bn3(self.conv3(outA + outB)), inplace=True)
        return out

    def initialize(self):
        weight_init(self)


# =>not used
class CBR(nn.Module):
    def __init__(self):
        super(CBR, self).__init__()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        return x

    def initialize(self):
        weight_init(self)


# =>not used
class CBR128(nn.Module):
    def __init__(self):
        super(CBR128, self).__init__()
        self.conv = nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        return x

    def initialize(self):
        weight_init(self)


# =>not used
class Cat_Gate(nn.Module):
    def __init__(self):
        super(Cat_Gate, self).__init__()

    def forward(self, S, D, F):
        x = torch.cat((S, D, F ), 1)
        return x

    def initialize(self):
        weight_init(self)


# =>not used
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.Catgate2 = Cat_Gate()
        self.Catgate3 = Cat_Gate()
        self.Catgate4 = Cat_Gate()
        self.Catgate5 = Cat_Gate()
        self.conv2 = nn.Conv2d(64 * 3, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64 * 3, 64, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64 * 3, 64, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.conv5 = nn.Conv2d(64 * 3, 64, kernel_size=3, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(64)
        self.fpn2 = FPN()
        self.fpn3 = FPN()
        self.fpn4 = FPN_()
        self.Cat = CAT()
        self.PFAN = PFAN()
        self.conv = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64)

    def forward(self,
                out2s, out3s, out4s, out5s,
                out2d, out3d, out4d, out5d,
                out2f, out3f, out4f, out5f):
        out2 = self.Catgate2(out2s, out2d, out2f)
        out2 = F.relu(self.bn2(self.conv2(out2)), inplace=True)
        out3 = self.Catgate3(out3s, out3d, out3f)
        out3 = F.relu(self.bn3(self.conv2(out3)), inplace=True)
        out4 = self.Catgate4(out4s, out4d, out4f)
        out4 = F.relu(self.bn4(self.conv4(out4)), inplace=True)
        out5 = self.Catgate5(out5s, out5d, out5f)
        out5 = F.relu(self.bn5(self.conv5(out5)), inplace=True)
        fpn4 = self.fpn4(out4, out5)
        fpn3 = self.fpn3(out3, fpn4)
        fpn2 = self.fpn2(out2, fpn3)
        final = fpn2

        return final

    def initialize(self):
        weight_init(self)


# =>not used
class DecoderS(nn.Module):
    def __init__(self):
        super(DecoderS, self).__init__()
        self.fpn2 = FPN_sig()
        self.fpn3 = FPN()
        self.fpn4 = FPN()

    def forward(self, out2s, out3s, out4s, out5s):
        fpn4 = self.fpn4(out4s, out5s)
        fpn3 = self.fpn3(out3s, fpn4)
        fpn2 = self.fpn2(out2s, fpn3)
        final = fpn2

        return final

    def initialize(self):
        weight_init(self)


# Conv
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
        if self.relu is not None:
            x = self.relu(x)
        return x

    def initialize(self):
        weight_init(self)


# =>one part of Attention
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    def initialize(self):
        weight_init(self)


# =>Attention
class CASA(nn.Module):
    def __init__(self):
        super(CASA, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 8),
            nn.ReLU(),
            # nn.BatchNorm1d(64//8),
            nn.Linear(64 // 8, 64)
        )

        self.conv1 = nn.Conv2d(64, 16, 1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.CB3 = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # Channel Attention
        avg_pool = F.avg_pool2d(x, x.size(2), stride=x.size(2))
        avg_pool = self.mlp(avg_pool)
        ca = avg_pool.unsqueeze(2).unsqueeze(3).expand_as(x)

        # Spatial Attention
        sa = self.conv1(x)
        sa = self.bn1(sa)
        sa = F.relu((sa), inplace=True)
        sa = self.conv2(sa)
        sa = self.bn2(sa)
        sa = F.relu((sa), inplace=True)
        sa = self.CB3(sa).expand_as(x)
        casa = F.sigmoid(ca * sa)
        return casa

    def initialize(self):
        weight_init(self)
        
        
# =>PAM
class PAM(nn.Module):
    def __init__(self):
        super(PAM, self).__init__()
        kernel_size = 3
        self.CASA = CASA()
        self.Atrous3 = BasicConv(64, 64, kernel_size, stride=1, padding=3, dilation=3, relu=False)
        self.Atrous5 = BasicConv(64, 64, kernel_size, stride=1, padding=5, dilation=5, relu=False)
        self.Atrous7 = BasicConv(64, 64, kernel_size, stride=1, padding=7, dilation=7, relu=False)
        self.Atrous9 = BasicConv(64, 64, kernel_size, stride=1, padding=9, dilation=9, relu=False)

        self.convP1 = BasicConv(64*4, 64, kernel_size, stride=1, padding=1, dilation=1, relu=True)
        self.convP2 = BasicConv(64*4, 64, kernel_size, stride=1, padding=1, dilation=1, relu=True)
        self.convC1 = BasicConv(64*3, 64, kernel_size, stride=1, padding=1, dilation=1, relu=True)
        self.convC2 = BasicConv(64*3, 64, kernel_size, stride=1, padding=1, dilation=1, relu=True)

    def forward(self, L, M, H):
        if L.size()[2:] != M.size()[2:]:
            L = F.interpolate(L, size=M.size()[2:], mode='bilinear')
            H = F.interpolate(H, size=M.size()[2:], mode='bilinear')
        O = M
        # =>SPU
        # D3
        x3 = self.Atrous3(M)
        x3 = F.sigmoid(x3)
        # D5
        x5 = self.Atrous5(M)
        x5 = F.sigmoid(x5)
        # D7
        x7 = self.Atrous7(M)
        x7 = F.sigmoid(x7)
        # D9
        x9 = self.Atrous9(M)
        x9 = F.sigmoid(x9)

        # => SCU
        P1 = torch.cat((x3, x5, x7, L), 1)
        P2 = torch.cat((x5, x7, x9, H), 1)
        P1 = self.convP1(P1)
        P2 = self.convP2(P2)
        # Densely Conected Structure 1
        C1 = torch.cat((P1, P2, O), 1)
        C1 = self.convC1(C1)
        # Attention
        casa = self.CASA(C1)
        # Densely Conected Structure 2
        C2 = torch.cat((casa*C1, C1, O), 1)
        C3 = self.convC2(C2)
        return C3

    def initialize(self):
        weight_init(self)


# =>not used
class FAM_sig(nn.Module):
    def __init__(self):
        super(FAM_sig, self).__init__()

        kernel_size = 3
        self.conv1x1 = BasicConv(64, 64, 1, stride=1, relu=False)
        # self.Atrous1 = BasicConv(64, 64, kernel_size, stride=1, padding=1, dilation=1, relu=False)
        self.Atrous3 = BasicConv(64, 64, kernel_size, stride=1, padding=3, dilation=3, relu=False)
        self.Atrous5 = BasicConv(64, 64, kernel_size, stride=1, padding=5, dilation=5, relu=False)
        self.Atrous7 = BasicConv(64, 64, kernel_size, stride=1, padding=7, dilation=7, relu=False)

        # self.conv1 = BasicConv(64, 64, kernel_size, stride=1, padding=1, dilation=1, relu=True)
        self.convP1 = BasicConv(64 * 4, 64, kernel_size, stride=1, padding=1, dilation=1, relu=True)
        self.convP2 = BasicConv(64 * 4, 64, kernel_size, stride=1, padding=1, dilation=1, relu=True)
        self.convC2 = BasicConv(64 * 2, 64, kernel_size, stride=1, padding=1, dilation=1, relu=False)

    def forward(self, M, H):
        if H.size()[2:] != M.size()[2:]:
            # L = F.interpolate(L, size=M.size()[2:], mode='bilinear')
            H = F.interpolate(H, size=M.size()[2:], mode='bilinear')
        x1 = self.conv1x1(M)
        x1 = F.sigmoid(x1)
        x3 = self.Atrous3(M)
        x3 = F.sigmoid(x3)
        x5 = self.Atrous5(M)
        x5 = F.sigmoid(x5)
        x7 = self.Atrous7(M)
        x7 = F.sigmoid(x7)
        P1 = torch.cat((x1, x3, x5, H), 1)
        P2 = torch.cat((x3, x5, x7, H), 1)
        P1 = self.convP1(P1)
        P2 = self.convP2(P2)
        # C1 = L * P1
        C1 = P1
        C2 = torch.cat((C1, P2), 1)
        C2 = self.convC2(C2)
        C2 = F.sigmoid(C2)
        return C2

    def initialize(self):
        weight_init(self)


# Decoder(FPN-PAM)
class FPN_PAM(nn.Module):
    def __init__(self):
        super(FPN_PAM, self).__init__()
        # PAM
        self.PAM4 = PAM()
        # one part of FPN
        self.fpn3 = FPN()
        # one part of FPN
        self.fpn2 = FPN_sig()

    def forward(self, out2s, out3s, out4s, out5s):
        iam4 = self.PAM4(out3s, out4s, out5s) # PAM
        iam3 = self.fpn3(out3s, iam4) # one part of FPN
        iam2 = self.fpn2(out2s, iam3) # one part of FPN
        final = iam2
        return final

    def initialize(self):
        weight_init(self)


# =>not used
class DecoderD(nn.Module):
    def __init__(self):
        super(DecoderD, self).__init__()
        self.fpn2 = FPN_sig()
        self.fpn3 = FPN()
        self.fpn4 = FPN()

    def forward(self, out2d, out3d, out4d, out5d):
        fpn4 = self.fpn4(out4d, out5d)
        fpn3 = self.fpn3(out3d, fpn4)
        fpn2 = self.fpn2(out2d, fpn3)
        final = fpn2
        return final

    def initialize(self):
        weight_init(self)


# ==========><==========
class ACDNet(nn.Module): 
    def __init__(self, cfg):
        super(ACDNet, self).__init__()
        self.cfg      = cfg
        self.Encoder_R = ResNet()  # Encoder-R
        # pre-processing for depth input
        self.conv1    = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)  # todo depth
        self.bn1      = nn.BatchNorm2d(64)

        # no used
        self.conv = nn.Conv2d(64*3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # todo depth
        self.bn = nn.BatchNorm2d(64)

        self.Encoder_D = ResNet_D()  # Encoder-D
        self.Encoder_F = ResNet_3()  # DCS

        # squeeze channel to 64
        self.squeeze5s = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4s = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3s = nn.Sequential(nn.Conv2d( 512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2s = nn.Sequential(nn.Conv2d( 256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.squeeze5d = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4d = nn.Sequential(nn.Conv2d(1024, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3d = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze2d = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # squeeze channel to 64
        self.squeeze5sf = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64))
        self.squeeze4sf = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256))  # 256
        self.squeeze3sf = nn.Sequential(nn.Conv2d(512, 128, 1), nn.BatchNorm2d(128))  # 128
        self.squeeze2sf = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64))  # 64
        self.squeeze5df = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64))
        self.squeeze4df = nn.Sequential(nn.Conv2d(1024, 256, 1), nn.BatchNorm2d(256))  # 256
        self.squeeze3df = nn.Sequential(nn.Conv2d(512, 128, 1), nn.BatchNorm2d(128))  # 128
        self.squeeze2df = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64))  # 64
        # squeeze channel to 64
        self.squeeze2F = nn.Sequential(nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze3F = nn.Sequential(nn.Conv2d(256, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze4F = nn.Sequential(nn.Conv2d(512, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        self.squeeze5F = nn.Sequential(nn.Conv2d(2048, 64, 1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        #  For ACG weights
        self.squeezeG1r = Gate(2048*2)
        self.squeezeG1d = Gate(2048*2)
        self.squeezeG2r = Gate(2048 * 2)
        self.squeezeG2d = Gate(2048 * 2)
        self.squeezeG3r = Gate(2048 * 2)
        self.squeezeG3d = Gate(2048 * 2)
        # Decoder
        self.decoderS = FPN_PAM()  # Decoder-R(FPN-PAM)
        self.decoderD = FPN_PAM()  # Decoder-D(FPN-PAM)
        self.decoderF = FPN_PAM()  # Decoder-F(FPN-PAM)

        self.linearr2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)  # for output
        self.linearr3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)  # no used
        self.linearr4 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)  # no used
        self.linearr5 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)   # no used

        self.initialize()

    def forward(self, x, depth, shape=None):  # RGB-D: x is RGB img, depth is D img
        # <=== === === Encoder === === ===>
        # =>bkbone for RGB
        # side outputs of Encoder-R: Sr1, Sr2, Sr3, Sr4, Sr5 
        Sr1, Sr2, Sr3, Sr4, Sr5 = self.Encoder_R(x)  # ResNet()
        
        # =>bkbone for Depth
        depth = F.relu(self.bn1(self.conv1(depth)), inplace=True)
        # side outputs of Encoder-D: Sd1, Sd2, Sd3, Sd4, Sd5 
        Sd1, Sd2, Sd3, Sd4, Sd5 = self.Encoder_D(depth)  # ResNet()

        # =>squeeze for side outputs

        # channel from 2048, 1024, 512, 256 to 64
        # R2, R3, R3, R4, R5, activation function is ReLU, for decoder-R
        R2, R3, R4, R5 = self.squeeze2s(Sr2), self.squeeze3s(Sr3), self.squeeze4s(Sr4), self.squeeze5s(Sr5)

        # R2', R3', R4', activation function is Sigmoid, for DCS
        R2_, R3_, R4_ = self.squeeze2sf(Sr2), self.squeeze3sf(Sr3), self.squeeze4sf(Sr4)
        R2_, R3_, R4_ = torch.sigmoid(R2_), torch.sigmoid(R3_), torch.sigmoid(R4_)

        # channel from 2048, 1024, 512, 256 to 64
        # D2, D3, D4, D5, activation function is ReLU, for decoder-D
        D2, D3, D4, D5 = self.squeeze2d(Sd2), self.squeeze3d(Sd3), self.squeeze4d(Sd4), self.squeeze5d(Sd5)

        # D2', D3', D4', activation function is Sigmoid, for DCS
        D2_, D3_, D4_ = self.squeeze2df(Sd2), self.squeeze3df(Sd3), self.squeeze4df(Sd4)
        D2_, D3_, D4_ = torch.sigmoid(D2_), torch.sigmoid(D3_), torch.sigmoid(D4_)

        # =>Decoder-R(FPN-PAM)
        R = self.decoderS(R2, R3, R4, R5)
        # =>Decoder-D(FPN-PAM)
        D = self.decoderD(D2, D3, D4, D5)
        
        # =>For ACG: cat(Sr5, Sd5)
        RD = torch.cat((Sr5, Sd5), 1)
        # generate weight
        G1r = self.squeezeG1r(RD)  # G1r
        G1d = self.squeezeG1d(RD)  # G1d
        G2r = self.squeezeG2r(RD)  # G2r
        G2d = self.squeezeG2d(RD)  # G2d
        G3r = self.squeezeG3r(RD)  # G3r
        G3d = self.squeezeG3d(RD)  # G3d

        # => DCS
        # inputs: R,    D,    R2',    D2',    R3',    D3',    R4',    D4'
        # outputs of DCS: out2sf, out2df, out3, out4, out5
        R2_, D2_, out3, out4, out5 = self.Encoder_F(R, D, R2_, D2_, R3_, D3_, R4_, D4_, G1r, G1d, G2r, G2d, G3r, G3d)
        # inputs: cat(out2sf, out2df)， out3, out4, out5 are outputs of DCS
        # outputs: out2F, out3F, out4F, out5F
        #  are     F2,    F3,    F4,    F5
        F2, F3, F4, F5 = self.squeeze2F(torch.cat((R2_, D2_), 1)), self.squeeze3F(out3), self.squeeze4F(out4), self.squeeze5F(out5)
        #                     128-64                                    256-64                512-64             2048-64
        # Decoder-F(FPN-PAM)
        # output: outF is F
        outF = self.decoderF(F2, F3, F4, F5)

        shape = x.size()[2:] if shape is None else shape  # todo noresize
        out = F.interpolate(self.linearr2(outF), size=shape, mode='bilinear')
        return out, G1r, G1d, G2r, G2d, G3r, G3d

    def initialize(self):
        if self.cfg.snapshot:
            self.load_state_dict(torch.load(self.cfg.snapshot))
        else:
            weight_init(self)
