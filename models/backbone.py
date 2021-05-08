from utils import Mish, Conv
from torch import nn
import torch


class ResBlock(nn.Module):
    """ Residual block that repeat 2 convolutional layers n times """
    def __init__(self, filters, num_blocks):
        super(ResBlock, self).__init__()
        self.module_list = nn.ModuleList()
        for i in range(num_blocks):
            res_block = nn.ModuleList()
            res_block.append(Conv(filters, filters, 1, 1, Mish()))
            res_block.append(Conv(filters, filters, 3, 1, Mish()))
            self.module_list.append(res_block)

    def forward(self, x):
        shortcut = x
        for module in self.module_list:
            shortcut = x
            for res in module: # noqa
                shortcut = res(shortcut)
            x = x + shortcut
        return shortcut


class DownSample(nn.Module):
    """ 'Legs' of CSPDarknet53 (first downsample block) """
    def __init__(self):
        super(DownSample, self).__init__()
        self.conv0 = Conv(3, 32, 3, 1, Mish())

        self.conv1 = Conv(32, 64, 3, 2, Mish())
        self.conv2 = Conv(64, 64, 1, 1, Mish())
        self.conv3 = Conv(64, 64, 1, 1, Mish())

        self.conv4 = Conv(64, 32, 1, 1, Mish())
        self.conv5 = Conv(32, 64, 3, 1, Mish())

        self.conv6 = Conv(64, 64, 1, 1, Mish())
        self.conv7 = Conv(128, 64, 1, 1, Mish())

    def forward(self, x):
        x = self.conv0(x)

        x = self.conv1(x)
        route = self.conv2(x)

        shortcut = self.conv3(x)
        x = self.conv4(shortcut)
        x = self.conv5(x)

        x = x + shortcut

        x = self.conv6(x)
        x = torch.cat([x, route], dim=1)
        x = self.conv7(x)
        return x


class CSPDarknetBlock(nn.Module):
    """ CSPDarknetBlock downsample block """
    def __init__(self, filters, num_blocks):
        super(CSPDarknetBlock, self).__init__()
        self.conv0 = Conv(filters, filters*2, 3, 2, Mish())
        self.conv1 = Conv(filters*2, filters, 1, 1, Mish())
        self.conv2 = Conv(filters*2, filters, 1, 1, Mish())
        self.res_block = ResBlock(filters, num_blocks)
        self.conv3 = Conv(filters, filters, 1, 1, Mish())
        self.conv4 = Conv(filters*2, filters*2, 1, 1, Mish())

    def forward(self, x):
        x = self.conv0(x)
        route = self.conv1(x)

        x = self.conv2(x)
        x = self.res_block(x)
        x = self.conv3(x)

        x = torch.cat([x, route], dim=1)
        x = self.conv4(x)
        return x


class CSPDarknet53(nn.Module):
    """  Backbone of YOLOv4 model """
    def __init__(self):
        super(CSPDarknet53, self).__init__()
        self.block0 = DownSample()
        self.block1 = CSPDarknetBlock(64, 2)
        self.block2 = CSPDarknetBlock(128, 8)
        self.block3 = CSPDarknetBlock(256, 8)
        self.block4 = CSPDarknetBlock(512, 4)

    def forward(self, x):
        x = self.block0(x)
        x = self.block1(x)
        output1 = self.block2(x)
        output2 = self.block3(output1)
        output3 = self.block4(output2)
        return output1, output2, output3
