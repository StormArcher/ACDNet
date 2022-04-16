import torch
import torch.nn as nn

# from module.BaseBlocks import BasicConv2d, BasicConv2d_sigmoid
# from utils.tensor_ops import cus_sample, upsample_add
# from backbone.origin.from_origin import Backbone_ResNet50_in3, Backbone_VGG16_in3
# from module.MyModule import AIM, SIM
import torch.nn.functional as F
from Gate_Library import GL_CFE


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


# ==========attention==========\
# ----------CASA-attention----------
class CASA(nn.Module):
    def __init__(self, channel=64, reduction_ratio=16):
        super(CASA, self).__init__()
        self.CA = CA(channel, reduction_ratio)
        self.SA = SA()

    def forward(self, x):
        CA = self.CA(x)
        out_CA = x * CA
        SA = self.SA(out_CA)
        out_SA = out_CA * SA
        return out_SA

    def initialize(self):
        weight_init(self)


# ----------CA-attention----------
class CA(nn.Module):
    def __init__(self, channel=64, reduction_ratio=16):
        super(CA, self).__init__()
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(channel, channel // reduction_ratio),  # 64->4
            nn.ReLU(),
            nn.Linear(channel // reduction_ratio, channel)
        )

    def forward(self, x):  # x.shape 0123 BCWH
        # x->pooling->mlp-x1
        avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # BCWH->BC
        avg_pool = self.mlp(avg_pool)  # BC
        # x->pooling->mlp-x2
        max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # BCWH->BC
        max_pool = self.mlp(max_pool)  # BC
        # x1 + x2
        att_sum = avg_pool + max_pool
        CA = torch.sigmoid(att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)  # BC->BC11->BCWH
        return CA

    def initialize(self):
        weight_init(self)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    def initialize(self):
        weight_init(self)


# ---------SA-attention---------
class SA(nn.Module):
    def __init__(self):
        super(SA, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(
            kernel_size-1) // 2, relu=False)

    def forward(self, x):  # x.shape 0123 BCWH
        x_compress = self.compress(x)  # BCWH->B2WH
        SA = self.spatial(x_compress)  # B2WH->B1WH  7x7conv
        SA = torch.sigmoid(SA)  # broadcasting
        return SA

    def initialize(self):
        weight_init(self)


class ChannelPool(nn.Module):  # BCWH->BCW->B1WH, BCWH->BCW->B1WH
    def forward(self, x):  # cat( BCWH->BCW->B1WH, BCWH->BCW->B1WH)  ->  B2WH
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
    
    def initialize(self):
        weight_init(self)
# ===========attention=========/


# ==========Clustering-attention==========\
class CLA(nn.Module):
    def __init__(self, channel=64, reduction_ratio=16):
        super(CLA, self).__init__()
        self.mlp1 = nn.Sequential(
            Flatten(),
            # nn.Linear(channel, channel // reduction_ratio),  # 64->4
            nn.Linear(channel*2, 1),  # 64->4
            # nn.ReLU()
            # nn.Linear(channel // reduction_ratio, channel)
        )
        self.mlp2 = nn.Sequential(
            Flatten(),
            # nn.Linear(channel, channel // reduction_ratio),  # 64->4
            nn.Linear(channel*3, 1),  # 64->4
            #nn.ReLU()
            # nn.Linear(channel // reduction_ratio, channel)
        )
        self.mlp3 = nn.Sequential(
            Flatten(),
            # nn.Linear(channel, channel // reduction_ratio),  # 64->4
            nn.Linear(channel*2, 1),  # 64->4
            # nn.ReLU()
            # nn.Linear(channel // reduction_ratio, channel)
        )

    def forward(self, c1, x1, x3, x5, x7, AvgMax):  # x.shape 0123 BCWH



        A = torch.cat((c1, x1), 1)
        B = torch.cat((x3, x5, x7), 1)
        C = AvgMax

        # x->pooling->mlp-x1
        A = F.avg_pool2d(A, (A.size(2), A.size(3)), stride=(A.size(2), A.size(3)))  # BCWH->BC
        B = F.avg_pool2d(B, (B.size(2), B.size(3)), stride=(B.size(2), B.size(3)))  # BCWH->BC
        C = F.avg_pool2d(C, (C.size(2), C.size(3)), stride=(C.size(2), C.size(3)))  # BCWH->BC

        A = self.mlp1(A)  # BC
        B = self.mlp2(B)  # BC
        C = self.mlp3(C)  # BC


        A = torch.sigmoid(A).unsqueeze(2).unsqueeze(3).expand_as(x1)  # BC->BC11->BCWH
        B = torch.sigmoid(B).unsqueeze(2).unsqueeze(3).expand_as(x1)  # BC->BC11->BCWH
        C = torch.sigmoid(C).unsqueeze(2).unsqueeze(3).expand_as(AvgMax)  # BC->BC11->BCWH


        return A, B, C

    def initialize(self):
        weight_init(self)


class AT(nn.Module):
    def __init__(self):
        super(AT, self).__init__()
        self.gate_library = GL_CFE(64)

        kernel_size = 3
        self.conv_cat = BasicConv(64*5, 64, kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, relu=False)
        self.bn_cat = nn.BatchNorm2d(64)

        self.conv1x1 = BasicConv(64, 64, 1, stride=1, relu=False)
        self.conv3 = BasicConv(64, 64, kernel_size, stride=1, padding=1, dilation=1, relu=False)

        self.conv = BasicConv(64, 64, 3, stride=1, padding=(3 - 1) // 2, dilation=1, relu=False)
        self.bn = nn.BatchNorm2d(64)

        k = 7
        self.conv2t1 = BasicConv(2, 1, 7, stride=1, padding=(7 - 1) // 2, dilation=1, relu=False)
        self.bn2t1 = nn.BatchNorm2d(1)

        # BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

        self.Atrous3 = BasicConv(64, 64, kernel_size, stride=1, padding=3, dilation=3, relu=False)
        self.Atrous5 = BasicConv(64, 64, kernel_size, stride=1, padding=5, dilation=5, relu=False)
        self.Atrous7 = BasicConv(64, 64, kernel_size, stride=1, padding=7, dilation=7, relu=False)

        # self.conv_avg = BasicConv(64, 64, kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, relu=False)
        self.conv_avgmax = BasicConv(64*2, 64*2, kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, relu=False)

        # self.conv_res = BasicConv(64, 64, 1, stride=1, relu=False)
        self.conv_final = BasicConv(64, 64, kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, relu=False)

        self.casa_x1 = CASA(64)
        self.casa_x3 = CASA(64)
        self.casa_x5 = CASA(64)
        self.casa_x7 = CASA(64)
        self.casa_Avg = CASA(64)
        self.casa_Max = CASA(64)

        self.cla = CLA()

        self.mlp1 = nn.Sequential(
            Flatten(),
            nn.Linear(64*5, 64*5 // 16),
            nn.ReLU(),
            nn.Linear(64 * 5 // 16, 1)

        )

        self.mlp21 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )
        self.mlp22 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )
        self.mlp23 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )
        self.mlp24 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )
        self.mlp25 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )

        self.conv5t1 = BasicConv(64*5, 64, 1, stride=1, padding=1, dilation=1, relu=False)
        self.bn5t1 = nn.BatchNorm2d(64)

    def forward(self, x):
        # print('---x',x.shape)
        x_ = x
        # Ga, Gb, Gc = self.gate_library(x)
        c1 = self.conv1x1(x)
        c1 = F.sigmoid(c1)  # F.sigmoid(x1)

        x1 = self.conv3(x)
        x1 = F.sigmoid(x1)

        # x1 = self.casa_x1(x1)
        x3 = self.Atrous3(x)
        x3 = F.sigmoid(x3)
        # x3 = self.casa_x3(x3)

        x5 = self.Atrous5(x)
        x5 = F.sigmoid(x5)
        # x5 = self.casa_x5(x5)
        x7 = self.Atrous7(x)
        x7 = F.sigmoid(x7)
        # x7 = self.casa_x7(x7)


        c1_ = F.avg_pool2d(c1, (c1.size(2), c1.size(3)), stride=(c1.size(2), c1.size(3)))
        x1_ = F.avg_pool2d(x1, (x1.size(2), x1.size(3)), stride=(x1.size(2), x1.size(3)))
        x3_ = F.avg_pool2d(x3, (x3.size(2), x3.size(3)), stride=(x3.size(2), x3.size(3)))
        x5_ = F.avg_pool2d(x5, (x5.size(2), x5.size(3)), stride=(x5.size(2), x5.size(3)))
        x7_ = F.avg_pool2d(x7, (x7.size(2), x7.size(3)), stride=(x7.size(2), x7.size(3)))
        w1 = self.mlp21(c1_)
        w2 = self.mlp22(x1_)
        w3 = self.mlp23(x3_)
        w4 = self.mlp24(x5_)
        w5 = self.mlp25(x7_)
        w1 = torch.sigmoid(w1).unsqueeze(2).unsqueeze(3).expand_as(x)
        w2 = torch.sigmoid(w2).unsqueeze(2).unsqueeze(3).expand_as(x)
        w3 = torch.sigmoid(w3).unsqueeze(2).unsqueeze(3).expand_as(x)
        w4 = torch.sigmoid(w4).unsqueeze(2).unsqueeze(3).expand_as(x)
        w5 = torch.sigmoid(w5).unsqueeze(2).unsqueeze(3).expand_as(x)

        x = torch.sigmoid(self.bn_cat(self.conv_cat(torch.cat((c1*w1, x1*w2, x3*w3, x5*w4, x7*w5), 1))))



        return x

    def initialize(self):
        weight_init(self)


class CFE(nn.Module):
    def __init__(self):
        super(CFE, self).__init__()
        self.gate_library = GL_CFE(64)

        kernel_size = 3
        self.conv_cat = BasicConv(64*5, 64, kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, relu=False)
        self.bn_cat = nn.BatchNorm2d(64)

        self.conv1x1 = BasicConv(64, 64, 1, stride=1, relu=False)
        self.conv3 = BasicConv(64, 64, kernel_size, stride=1, padding=1, dilation=1, relu=False)

        self.conv = BasicConv(64, 64, 3, stride=1, padding=(3 - 1) // 2, dilation=1, relu=False)
        self.bn = nn.BatchNorm2d(64)

        k = 7
        self.conv2t1 = BasicConv(2, 1, 7, stride=1, padding=(7 - 1) // 2, dilation=1, relu=False)
        self.bn2t1 = nn.BatchNorm2d(1)

        # BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

        self.Atrous3 = BasicConv(64, 64, kernel_size, stride=1, padding=3, dilation=3, relu=False)
        self.Atrous5 = BasicConv(64, 64, kernel_size, stride=1, padding=5, dilation=5, relu=False)
        self.Atrous7 = BasicConv(64, 64, kernel_size, stride=1, padding=7, dilation=7, relu=False)

        # self.conv_avg = BasicConv(64, 64, kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, relu=False)
        self.conv_avgmax = BasicConv(64*2, 64*2, kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, relu=False)

        # self.conv_res = BasicConv(64, 64, 1, stride=1, relu=False)
        self.conv_final = BasicConv(64, 64, kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1, relu=False)

        self.casa_x1 = CASA(64)
        self.casa_x3 = CASA(64)
        self.casa_x5 = CASA(64)
        self.casa_x7 = CASA(64)
        self.casa_Avg = CASA(64)
        self.casa_Max = CASA(64)

        self.cla = CLA()

        self.mlp1 = nn.Sequential(
            Flatten(),
            nn.Linear(64*5, 64*5 // 16),
            nn.ReLU(),
            nn.Linear(64 * 5 // 16, 1)

        )

        self.mlp21 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )
        self.mlp22 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )
        self.mlp23 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )
        self.mlp24 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )
        self.mlp25 = nn.Sequential(
            Flatten(),
            nn.Linear(64, 64 // 16),
            nn.ReLU(),
            nn.Linear(64 // 16, 1)
        )

        self.conv5t1 = BasicConv(64*5, 64, 1, stride=1, padding=1, dilation=1, relu=False)
        self.bn5t1 = nn.BatchNorm2d(64)

    def forward(self, x):
        # print('---x',x.shape)
        x_ = x
        # Ga, Gb, Gc = self.gate_library(x)
        c1 = self.conv1x1(x)
        c1 = F.sigmoid(c1)  # F.sigmoid(x1)

        x1 = self.conv3(x)
        x1 = F.sigmoid(x1)

        # x1 = self.casa_x1(x1)
        x3 = self.Atrous3(x)
        x3 = F.sigmoid(x3)
        # x3 = self.casa_x3(x3)

        x5 = self.Atrous5(x)
        x5 = F.sigmoid(x5)
        # x5 = self.casa_x5(x5)
        x7 = self.Atrous7(x)
        x7 = F.sigmoid(x7)
        # x7 = self.casa_x7(x7)

        # Avg = F.avg_pool2d(x, (2, 2), stride=(2, 2))
        # Avg = F.sigmoid(self.conv_avg(Avg))
        # Avg = F.interpolate(Avg, size=x.size()[2:], mode='bilinear')

        # Max = F.max_pool2d(x, (2, 2), stride=(2, 2))
        # AvgMax = F.sigmoid(self.conv_avgmax( torch.cat((Max, Avg), 1) ))

        # ==> SA
        # Smax_Savg = torch.cat((torch.max(AvgMax, 1)[0].unsqueeze(1), torch.mean(AvgMax, 1).unsqueeze(1)), dim=1)
        # BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        # Smax_Savg = self.conv2t1(Smax_Savg)
        # Smax_Savg = torch.sigmoid(Smax_Savg)
        # AvgMax = Smax_Savg * AvgMax

        # AvgMax = F.interpolate(AvgMax, size=x.size()[2:], mode='bilinear')

        # A, B, C = self.cla(c1, x1, x3, x5, x7, AvgMax)

        # x = torch.cat((c1 , x1 , x3 , x5 , x7), 1)


        # x = self.conv_cat(torch.cat((c1 , x1 , x3 , x5 , x7), 1))

        # ===>CA
        #avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
        # print('---avg', avg_pool.shape)

        # channel_sum = self.mlp1(avg_pool)

        c1_ = F.avg_pool2d(c1, (c1.size(2), c1.size(3)), stride=(c1.size(2), c1.size(3)))
        x1_ = F.avg_pool2d(x1, (x1.size(2), x1.size(3)), stride=(x1.size(2), x1.size(3)))
        x3_ = F.avg_pool2d(x3, (x3.size(2), x3.size(3)), stride=(x3.size(2), x3.size(3)))
        x5_ = F.avg_pool2d(x5, (x5.size(2), x5.size(3)), stride=(x5.size(2), x5.size(3)))
        x7_ = F.avg_pool2d(x7, (x7.size(2), x7.size(3)), stride=(x7.size(2), x7.size(3)))
        w1 = self.mlp21(c1_)
        w2 = self.mlp22(x1_)
        w3 = self.mlp23(x3_)
        w4 = self.mlp24(x5_)
        w5 = self.mlp25(x7_)
        w1 = torch.sigmoid(w1).unsqueeze(2).unsqueeze(3).expand_as(x)
        w2 = torch.sigmoid(w2).unsqueeze(2).unsqueeze(3).expand_as(x)
        w3 = torch.sigmoid(w3).unsqueeze(2).unsqueeze(3).expand_as(x)
        w4 = torch.sigmoid(w4).unsqueeze(2).unsqueeze(3).expand_as(x)
        w5 = torch.sigmoid(w5).unsqueeze(2).unsqueeze(3).expand_as(x)

        x = torch.sigmoid(self.bn_cat(self.conv_cat(torch.cat((c1*w1, x1*w2, x3*w3, x5*w4, x7*w5), 1))))

        # ===>5->1
        # x = torch.sigmoid(self.bn5t1(self.conv5t1(x)))  # 64*5->64
        # ===>SA
        # x_cat = F.relu(self.conv_cat(torch.cat((x1*Ga, x3*Ga, x5*Gb, x7*Gb, Avg*Gc, Max*Gc), 1)), inplace=True)
        #  cat(x1x3, x5x7, avgmax) + res
        # x_res = F.relu(self.conv_res(x), inplace=True)
        # x = F.relu(self.conv_final(x_cat + x), inplace=True)
        #print('---x', x.shape)
        Smax_Savg = torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        #print('---Smax_Savg1', Smax_Savg.shape)
        Smax_Savg = self.bn2t1(self.conv2t1(Smax_Savg))
        #print('---Smax_Savg2', Smax_Savg.shape)
        Smax_Savg = torch.sigmoid(Smax_Savg)

        x = Smax_Savg * x
        x = x + x_

        x = torch.sigmoid(self.bn(self.conv(x)))

        # x1 = (x1 * G1 * 4)/(G1+G3+G5+G7)
        # x3 = (x3 * G3 * 4)/(G1+G3+G5+G7)
        # x5 = (x5 * G5 * 4)/(G1+G3+G5+G7)
        # x7 = (x7 * G7 * 4)/(G1+G3+G5+G7)
        '''
        # a是继续写入, w是覆盖
        with open("/home/nk/zjc/PycharmProjects/1 CVPR/MINet_VGG_FPN+Atrous_wCAT/MINet-master/code/recordw1.txt",
                  "a") as f:
            f.write('\n' + str(G1[0][0][0][0]) + '\n' + str(G1[1][0][0][0]) + '\n' + str(G1[2][0][0][0]) + '\n' + str(
                G1[3][0][0][0]))  # a是继续写入, w是覆盖
        with open("/home/nk/zjc/PycharmProjects/1 CVPR/MINet_VGG_FPN+Atrous_wCAT/MINet-master/code/recordw3.txt",
                  "a") as f:
            f.write('\n' + str(G3[0][0][0][0]) + '\n' + str(G3[1][0][0][0]) + '\n' + str(G3[2][0][0][0]) + '\n' + str(
                G3[3][0][0][0]))  # a是继续写入, w是覆盖
        with open("/home/nk/zjc/PycharmProjects/1 CVPR/MINet_VGG_FPN+Atrous_wCAT/MINet-master/code/recordw5.txt",
                  "a") as f:
            f.write('\n' + str(G5[0][0][0][0]) + '\n' + str(G5[1][0][0][0]) + '\n' + str(G5[2][0][0][0]) + '\n' + str(
                G5[3][0][0][0]))  # a是继续写入, w是覆盖
        with open("/home/nk/zjc/PycharmProjects/1 CVPR/MINet_VGG_FPN+Atrous_wCAT/MINet-master/code/recordw7.txt",
                  "a") as f:
            f.write('\n' + str(G7[0][0][0][0]) + '\n' + str(G7[1][0][0][0]) + '\n' + str(G7[2][0][0][0]) + '\n' + str(
                G7[3][0][0][0]))  # a是继续写入, w是覆盖'''


        return x

    def initialize(self):
        weight_init(self)