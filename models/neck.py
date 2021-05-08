from utils import Conv
from torch import nn
import torch


class SPP(nn.Module):
    def __init__(self):
        super(SPP, self).__init__()
        filters = 512
        self.conv0 = Conv(filters*2, filters, 1, 1)
        self.conv1 = Conv(filters, filters*2, 3, 1)
        self.conv2 = Conv(filters*2, filters, 1, 1)

        self.maxpool1 = nn.MaxPool2d(5, stride=1, padding=5//2)
        self.maxpool2 = nn.MaxPool2d(9, stride=1, padding=9//2)
        self.maxpool3 = nn.MaxPool2d(13, stride=1, padding=13//2)

        self.conv3 = Conv(filters*4, filters, 1, 1)
        self.conv4 = Conv(filters, filters*2, 3, 1)
        self.conv5 = Conv(filters*2, filters, 1, 1)

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        x1 = self.maxpool1(x)
        x2 = self.maxpool2(x)
        x3 = self.maxpool3(x)
        x = torch.cat([x3, x2, x1, x], dim=1)

        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class CSPUp(nn.Module):
    def __init__(self, filters):
        super(CSPUp, self).__init__()
        self.conv0 = Conv(filters*2, filters, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2)
        self.conv1 = Conv(filters*2, filters, 1, 1)
        self.conv2 = Conv(filters*2, filters, 1, 1)
        self.conv3 = Conv(filters, filters*2, 3, 1)
        self.conv4 = Conv(filters*2, filters, 1, 1)
        self.conv5 = Conv(filters, filters*2, 3, 1)
        self.conv6 = Conv(filters*2, filters, 1, 1)

    def forward(self, x):
        a, b = x

        a = self.conv1(a)

        b = self.conv0(b)
        b = self.upsample(b)

        x = torch.cat([a, b], dim=1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)

        return x


class YoloNeck(nn.Module):
    def __init__(self):
        super(YoloNeck, self).__init__()
        self.spp = SPP()
        self.up1 = CSPUp(256)
        self.up2 = CSPUp(128)

    def forward(self, x):
        a, b, c = x
        c = self.spp(c)
        b = self.up1((b, c))
        a = self.up2((a, b))

        return a, b, c

