import torch
from collections import OrderedDict

from pth2pth.load_backbone import load_darknet_weights
from pth2pth.load_neck import load_neck_weights
from pth2pth.load_head import load_head_weights


def split_parts(state_dict):
    backbone_dict = OrderedDict()
    neck_dict = OrderedDict()
    head_dict = OrderedDict()

    for k in state_dict.keys():
        if k[:4] == 'down':
            backbone_dict[k] = state_dict[k]
        elif k[:4] == 'neek': # noqa
            neck_dict[k] = state_dict[k]
        elif k[:4] == 'head':
            head_dict[k] = state_dict[k]

    return backbone_dict, neck_dict, head_dict


def load_yolo(yolo_model):
    state_dict = torch.load("pth2pth/yolov.pth")
    backbone_dict, neck_dict, head_dict = split_parts(state_dict)

    load_darknet_weights(yolo_model.backbone, backbone_dict)
    load_neck_weights(yolo_model.neck, neck_dict)
    load_head_weights(yolo_model.head, head_dict)

    return backbone_dict, neck_dict, head_dict
