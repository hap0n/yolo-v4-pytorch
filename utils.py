import torch
from torch import nn


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x): # noqa
        return x * (torch.tanh(nn.Softplus()(x)))


class Conv(nn.Module):
    """ Convolutional layer with batch normalization and activation layer (LeakyReLU or Mish) """
    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 activation=nn.LeakyReLU(0.1), use_bn=True, bias=False):
        super(Conv, self).__init__()
        padding = (kernel_size - 1) // 2

        self.layers = nn.ModuleList()
        self.layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias))
        if use_bn:
            self.layers.append(nn.BatchNorm2d(out_channels))
        self.layers.append(activation)

    def forward(self, x):
        for layer in self.layers:
            if layer:
                x = layer(x)
        return x
