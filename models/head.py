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

    @staticmethod
    def boxes_regression(inputs, anchors):
        """
        Tensor.shape = [batch_size, channels, width, height]
        Channels = (coords(4) + objectness(1) + n_classes(80 in COCO dataset)) * n_anchor_boxes (3) = 255

        small_input, medium_input, large_input = inputs
        small_anchors, medium_anchors, large_anchors = anchors
        anchors.shape = [2, 1, 1, 3]
        """
        width, height = (inputs.shape[2], inputs.shape[3])

        # inputs: [batch_size, C, W, H] --> [batch_size, 5 + n_classes, W, H, n_anchors]
        inputs = torch.unsqueeze(inputs, dim=4)
        inputs = torch.split(inputs, 85, dim=1)
        inputs = torch.cat(inputs, dim=-1)

        # split center coords, width and height, objectness and class probabilities
        t_xy, t_wh, objectness, class_prob = torch.split(inputs, [2, 2, 1, N_CLASSES], dim=1)

        # create grid with shape: [2, W, H]
        grid = torch.meshgrid(torch.arange(width), torch.arange(height))
        grid = torch.stack(grid, dim=0)

        # apply sigmoid to center coords, objectness and class probabilities
        b_xy = torch.sigmoid(t_xy)
        objectness = torch.sigmoid(objectness)
        class_p = torch.sigmoid(class_prob)

        # normalize b_xy values to (0, 1)
        b_xy = (b_xy + grid.float()) / torch.tensor([width, height], dtype=torch.float32)

        # calculate real width and height of bounding boxes
        anchors = torch.unsqueeze(anchors, dim=1)
        anchors = torch.unsqueeze(anchors, dim=1)
        b_wh = torch.exp(t_wh) * anchors

        # calculate class confidences
        conf = class_p * objectness

        # boxes = x1, y1, x2, y2

        return conf


class YoloHead(nn.Module):
    def __init__(self, inference=False):
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

            a, b, c = a, b, c          # apply region boxes

        print(a.shape, b.shape, c.shape)
        return a, b, c
