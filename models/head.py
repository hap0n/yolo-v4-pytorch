from torch import nn
from utils import Conv
import torch


ANCHORS = torch.Tensor([
    [[12, 16], [19, 36], [40, 28]],
    [[36, 75], [76, 55], [72, 146]],
    [[142, 110], [192, 243], [459, 401]]
])
N_ANCHORS = 3
N_CLASSES = 80
OUT_CHANNELS = (5 + N_CLASSES) * N_ANCHORS


class YoloLayer(nn.Module):
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()
        self.anchors = anchors.view(1, N_ANCHORS, 1, 1, 2).float()

    def forward(self, inputs, img_size):
        batch_size, _, width, height = inputs.shape

        # inputs: [batch_size, C, W, H] --> [batch_size, n_anchors, W, H, 5 + n_classes]
        inputs = inputs.view(batch_size, N_ANCHORS, 5 + N_CLASSES, width, height).permute(0, 1, 3, 4, 2).contiguous()

        # split center coords, width and height, objectness and class probabilities
        t_xy, t_wh, objectness, class_prob = torch.split(inputs, [2, 2, 1, N_CLASSES], dim=-1)

        # apply sigmoid to objectness and class probabilities
        objectness = torch.sigmoid(objectness)
        class_prob = torch.sigmoid(class_prob)

        # create grid with shape: [W, H, 2]
        yv, xv = torch.meshgrid([torch.arange(width), torch.arange(height)])
        c_xy = torch.stack((xv, yv), dim=2).view(1, 1, width, height, 2).float()

        if inputs.is_cuda:
            c_xy = c_xy.cuda()
            self.anchors = self.anchors.cuda()

        stride = img_size // width
        b_xy = (torch.sigmoid(t_xy) + c_xy) * stride

        # calculate real width and height of bounding boxes
        b_wh = torch.exp(t_wh) * self.anchors

        x1y1 = b_xy - b_wh / 2.
        x2y2 = b_xy + b_wh / 2.

        boxes = torch.cat([x1y1, x2y2], dim=-1).view(batch_size, -1, 4)
        objectness = objectness.view(batch_size, -1, 1)
        class_prob = class_prob.view(batch_size, -1, N_CLASSES)

        prediction = torch.cat([boxes, objectness, class_prob], dim=-1)

        return prediction


class CSPDown(nn.Module):
    def __init__(self, filters):
        super(CSPDown, self).__init__()
        self.conv0 = Conv(filters, filters*2, 3, 2)
        self.conv1 = Conv(filters*4, filters*2, 1, 1)
        self.conv2 = Conv(filters*2, filters*4, 3, 1)
        self.conv3 = Conv(filters*4, filters*2, 1, 1)
        self.conv4 = Conv(filters*2, filters*4, 3, 1)
        self.conv5 = Conv(filters*4, filters*2, 1, 1)

    def forward(self, x):
        a, b = x

        a = self.conv0(a)
        x = torch.cat([a, b], dim=1)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)

        return x


class YoloHead(nn.Module):
    def __init__(self):
        super(YoloHead, self).__init__()

        self.conv1 = Conv(128, 256, 3, 1)
        self.conv2 = Conv(256, OUT_CHANNELS, 1, 1, activation=None, use_bn=False, bias=True)
        self.yolo1 = YoloLayer(ANCHORS[0])

        self.down1 = CSPDown(128)
        self.conv3 = Conv(256, 512, 3, 1)
        self.conv4 = Conv(512, OUT_CHANNELS, 1, 1, activation=None, use_bn=False, bias=True)
        self.yolo2 = YoloLayer(ANCHORS[1])

        self.down2 = CSPDown(256)
        self.conv5 = Conv(512, 1024, 3, 1)
        self.conv6 = Conv(1024, OUT_CHANNELS, 1, 1, activation=None, use_bn=False, bias=True)
        self.yolo3 = YoloLayer(ANCHORS[2])

    def forward(self, x, img_size, inference=True):
        a, b, c = x

        b = self.down1((a, b))
        c = self.down2((b, c))

        output1 = self.conv1(a)
        output1 = self.conv2(output1)

        output2 = self.conv3(b)
        output2 = self.conv4(output2)

        output3 = self.conv5(c)
        output3 = self.conv6(output3)

        if inference:
            prediction1 = self.yolo1(output1, img_size)
            prediction2 = self.yolo2(output2, img_size)
            prediction3 = self.yolo3(output3, img_size)

            prediction = torch.cat([prediction1, prediction2, prediction3], dim=1)

            return prediction

        return output1, output2, output3
