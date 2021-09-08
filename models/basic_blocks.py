import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# CBAM Block (to add attention)
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        x = out * x

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv1(torch.cat([avg_out, max_out], dim=1))
        out = self.sigmoid(out)
        x = out * x

        return x


# Depth-wise Separable Convolution
class SeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super(SeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=in_ch
        )
        self.point_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out


# Basic Block for Xception Network
class XceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num=3, conv=SeparableConv, cbam=False, pre_relu=False,
                 max_pool=False, skip=False, diy=None):
        super(XceptionBlock, self).__init__()
        self.skip = skip
        if diy is None:
            in_cs, out_cs = [in_channels] + [out_channels] * (conv_num - 1), [out_channels] * conv_num
        else:
            in_cs, out_cs = [in_channels] + diy[:-1], diy

        blocks = []
        for i in range(conv_num):
            blocks += [nn.ReLU(),
                       conv(in_cs[i], out_cs[i], 3, 1),
                       nn.BatchNorm2d(out_cs[i])]
        if not pre_relu:
            blocks.pop(0)
        if max_pool:
            blocks += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        if cbam:
            blocks += [CBAM(out_channels)]
        self.block = nn.Sequential(*blocks).to(device)

        # Resident Connections
        self.resConnection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2) if max_pool \
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.block(x) + self.resConnection(x) if self.skip else self.block(x)
        return out


# Basic Block for U-Net based Network
class UNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, max_pool=False, cbam=False, upsample=False):
        super(UNetBlock, self).__init__()
        blocks = []
        if max_pool:
            blocks += [nn.MaxPool2d(kernel_size=2, stride=2)]
        blocks += [conv(in_channels, out_channels, 3, 1),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(),
                   conv(out_channels, out_channels, 3, 1),
                   nn.BatchNorm2d(out_channels),
                   nn.ReLU(),
                   ]
        if cbam:
            blocks += [CBAM(out_channels)]
        if upsample:
            blocks += [nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2),
                       nn.BatchNorm2d(out_channels // 2),
                       nn.ReLU()
                       ]
        self.blocks = nn.Sequential(*blocks).to(device)

    def forward(self, x):
        return self.blocks(x)
