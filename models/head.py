from torch import nn
from utils import Conv
import torch


ANCHORS = [[[12, 16], [19, 36], [40, 28]],
           [[36, 75], [76, 55], [72, 146]],
           [[142, 110], [192, 243], [459, 401]]]
N_ANCHOR_BOXES = 3
N_CLASSES = 80
OUT_CHANNELS = (5 + N_CLASSES) * N_ANCHOR_BOXES


class YoloLayer(nn.Module):
    def __init__(self, anchors):
        super(YoloLayer, self).__init__()

        anchors = torch.tensor(anchors, dtype=torch.float32)
        anchors = torch.swapaxes(anchors, 0, 1)
        anchors = torch.unsqueeze(anchors, dim=1)
        anchors = torch.unsqueeze(anchors, dim=1)
        self.anchors = anchors

    def forward(self, inputs):
        # if self.training:
        #     return inputs
        x = self.boxes_regression(inputs)
        # nms
        return x

    def boxes_regression(self, inputs):
        """
        Split input flot32 tensor into t_xy, t_wh, objectness and class probabilities.
        Apply sigmoid function to t_x, t_y, objectness and class probability; apply exp
        function to t_w, t_h. Then add offset to center coordinates of bounding boxes
        and calculate x1, y1, x2 and y2 values to create float32 tensor with bounding
        boxes. Then calculate class probabilities and get scores from this tensor.
        :param
            inputs: A [batch_size, (5 + n_classes) * n_anchors, width, height] float32
              tensor containing output of last conv layers of YOLO model.
        :return:
            boxes: A [batch_size, n_boxes, 4] float32 tensor containing normalized
              x1, y1, x2, y2 values in range [0, 1] for bounding boxes.
            conf: A [batch_size, n_boxes, n_classes] float32 tensor containing
              probabilities of classes that correspond to bounding boxes.
            scores: A [batch_size, n_boxes] float32 tensor containing scores of
              objectness than correspond to bounding boxes.
        """
        batch_size, channels, width, height = inputs.shape

        # inputs: [batch_size, C, W, H] --> [batch_size, 5 + n_classes, W, H, n_anchors]
        inputs = torch.unsqueeze(inputs, dim=4)
        inputs = torch.split(inputs, 85, dim=1)
        inputs = torch.cat(inputs, dim=-1)

        # split center coords, width and height, objectness and class probabilities
        t_xy, t_wh, objectness, class_prob = torch.split(inputs, [2, 2, 1, N_CLASSES], dim=1)

        # create grid with shape: [2, W, H]
        grid = torch.meshgrid(torch.arange(width), torch.arange(height))
        grid = torch.stack(grid, dim=0).view(1, 2, width, height, 1)

        # apply sigmoid to center coords, objectness and class probabilities
        b_xy = torch.sigmoid(t_xy)
        objectness = torch.sigmoid(objectness)
        class_p = torch.sigmoid(class_prob)

        # normalize b_xy values to (0, 1)
        n = torch.tensor([width, height], dtype=torch.float32).view(1, -1, 1, 1, 1)
        b_xy = (b_xy + grid.float()) / n

        # calculate real width and height of bounding boxes
        b_wh = torch.exp(t_wh) * self.anchors

        # calculate class confidences
        conf = class_p * objectness
        conf = conf.view(batch_size, N_CLASSES, -1)
        conf = torch.swapaxes(conf, 1, 2)

        # create tensor with scores
        scores = torch.max(conf, dim=2).values

        # calculate x1, y1, x2, y2 and stack them
        x1y1 = b_xy - b_wh / 2
        x1, y1 = torch.split(x1y1, [1, 1], dim=1)

        x2y2 = b_xy + b_wh / 2
        x2, y2 = torch.split(x2y2, [1, 1], dim=1)

        boxes = torch.cat([x1, y1, x2, y2], dim=1).view(batch_size, 4, -1)
        boxes = torch.swapaxes(boxes, 1, 2)

        return boxes, conf, scores

    @staticmethod
    def yolo_nms():
        return


class YoloHead(nn.Module):
    def __init__(self, inference=False):
        super(YoloHead, self).__init__()
        self.inference = inference

        self.conv1 = Conv(128, 256, 3, 1)
        self.conv2 = Conv(256, OUT_CHANNELS, 1, 1, activation=None, use_bn=False, bias=True)
        self.yolo1 = YoloLayer(ANCHORS[0])

        self.conv3 = Conv(256, 512, 3, 1)
        self.conv4 = Conv(512, OUT_CHANNELS, 1, 1, activation=None, use_bn=False, bias=True)
        self.yolo2 = YoloLayer(ANCHORS[1])

        self.conv5 = Conv(512, 1024, 3, 1)
        self.conv6 = Conv(1024, OUT_CHANNELS, 1, 1, activation=None, use_bn=False, bias=True)
        self.yolo3 = YoloLayer(ANCHORS[2])

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
