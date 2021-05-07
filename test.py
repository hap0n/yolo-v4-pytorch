import torch
from models.head import YoloLayer

ANCHORS = [[[12, 16], [19, 36], [40, 28]],
           [[36, 75], [76, 55], [72, 146]],
           [[142, 110], [192, 243], [459, 401]]]

yolo_layer = YoloLayer(ANCHORS[0])
inputs = torch.zeros([32, 255, 64, 64])
boxes, conf, scores = yolo_layer(inputs)
print(boxes.shape, conf.shape, scores.shape)
