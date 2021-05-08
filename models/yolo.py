from models.backbone import CSPDarknet53
from models.neck import YoloNeck
from models.head import YoloHead

from torch import nn


class YOLOv4(nn.Module):
    def __init__(self):
        super(YOLOv4, self).__init__()

        self.backbone = CSPDarknet53()
        self.neck = YoloNeck()
        self.head = YoloHead()

    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        x = self.head(x)

        return x
