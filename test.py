import torch

x = torch.zeros([32, 2, 64, 64, 3], dtype=torch.float32)
anchors = torch.ones([2, 3], dtype=torch.float32)
z = x * anchors
print(z)
