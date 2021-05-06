import utils
from models.backbone import CSPDarknet53
from models.neck import PANet
import torch


x = torch.zeros([1, 3, 512, 512])
darknet = CSPDarknet53()
panet = PANet()
x = darknet(x)
x = panet(x)
o1, o2, o3 = x
print(o1.shape, o2.shape, o3.shape)


