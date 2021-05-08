import torch
from collections import OrderedDict


def split_parts(state_dict):
    backbone_dict = OrderedDict()
    neck_dict = OrderedDict()
    head_dict = OrderedDict()

    for k in state_dict.keys():
        if k[:4] == 'down':
            backbone_dict[k] = state_dict[k]
        elif k[:4] == 'neck':
            neck_dict[k] = state_dict[k]
        elif k[:4] == 'head':
            head_dict[k] = state_dict[k]

    return backbone_dict, neck_dict, head_dict


def load_yolo():
    state_dict = torch.load("pth2pth/yolov.pth")
    backbone_dict, neck_dict, head_dict = split_parts(state_dict)

    return backbone_dict, neck_dict, head_dict
