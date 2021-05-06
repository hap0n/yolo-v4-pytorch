from torch import nn
from utils import Conv
import torch


ANCHORS = [12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401]
N_ANCHOR_BOXES = 3
N_CLASSES = 80
OUT_CHANNELS = (5 + N_CLASSES) * N_ANCHOR_BOXES


class YoloLayer(nn.Module):
    def __init__(self, anchor_mask, stride, num_anchors=9):
        super(YoloLayer, self).__init__()
        pass

    def call(self, call):
        pass


class YoloHead(nn.Module):
    def __init__(self, inference=True):
        super(YoloHead, self).__init__()
        self.inference = inference

        self.conv1 = Conv(128, 256, 3, 1)
        self.conv2 = Conv(256, OUT_CHANNELS, 1, 1, activation=None, use_bn=False, bias=True)
        self.yolo1 = YoloLayer([0, 1, 2], 8)

        self.conv3 = Conv(256, 512, 3, 1)
        self.conv4 = Conv(512, OUT_CHANNELS, 1, 1, activation=None, use_bn=False, bias=True)
        self.yolo2 = YoloLayer([3, 4, 5], 16)

        self.conv5 = Conv(512, 1024, 3, 1)
        self.conv6 = Conv(1024, OUT_CHANNELS, 1, 1, activation=None, use_bn=False, bias=True)
        self.yolo3 = YoloLayer([6, 7, 8], 32)

    def forward(self, x):
        a, b, c = x

        a = self.conv1(a)
        a = self.conv2(a)

        b = self.conv3(b)
        b = self.conv4(b)

        c = self.conv5(c)
        c = self.conv6(c)

        if self.inference:
            a = self.yolo1(a)
            b = self.yolo2(b)
            c = self.yolo3(c)
            print(a.shape(), b.shape(), c.shape())
            a, b, c = a, b, c          # apply region boxes

        return a, b, c



