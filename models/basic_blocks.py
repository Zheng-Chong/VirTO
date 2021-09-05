import torch.nn as nn
import torch


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
    def __init__(self, in_ch, out_ch):
        super(SeparableConv, self).__init__()
        self.depth_conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=3,
            stride=1,
            padding=1,
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


# Basic Block for ResNet based Network
class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, conv_num=3, separable_conv=False, cbam=False,
                 pre_relu=True, max_pool=False, diy=None, skip=True, upsanmple=False):
        super(ResnetBlock, self).__init__()
        self.skip = skip
        blocks = []
        in_cs = [in_channels] + [out_channels] * (conv_num-1) if diy is None else diy
        for i in in_cs:
            conv_layer = SeparableConv(i, out_channels) if separable_conv else nn.Conv2d(i, out_channels, kernel_size=3, padding=1)
            blocks += [nn.ReLU(),
                       conv_layer,
                       nn.BatchNorm2d(out_channels)]
        if not pre_relu:
            blocks.pop(0)
        if max_pool:
            blocks += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        if cbam:
            blocks += [CBAM(out_channels)]
        if upsanmple:
            blocks += [nn.ReLU(),
                       nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2),
                       nn.BatchNorm2d(out_channels // 2)
                       ]

        self.block = nn.Sequential(*blocks)
        # Resident Connections
        self.resConnection = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2) if max_pool \
            else nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        out = self.block(x) + self.resConnection(x) if self.skip else self.block(x)
        return out


# Convolutional layer with l2 weight normalization and learned scaling parameters
class NormConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.beta = nn.Parameter(torch.zeros([1, out_channels, 1, 1], dtype=torch.float32))
        self.gamma = nn.Parameter(torch.ones([1, out_channels, 1, 1], dtype=torch.float32))
        self.conv = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            name="weight",
        )

    def forward(self, x):
        # weight normalization
        # self.conv.weight = normalize(self.conv.weight., dim=[0, 2, 3])
        out = self.conv(x)
        out = self.gamma * out + self.beta
        return out


class Downsample(nn.Module):
    def __init__(self, channels, out_channels=None, conv_layer=NormConv2d):
        super().__init__()
        if out_channels is None:
            self.down = conv_layer(channels, channels, kernel_size=3, stride=2, padding=1)
        else:
            self.down = conv_layer(channels, out_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels, subpixel=True, conv_layer=NormConv2d):
        super().__init__()
        if subpixel:
            self.up = conv_layer(in_channels, 4 * out_channels, 3, padding=1)
            self.op2 = DepthToSpace(block_size=2)
        else:
            # channels have to be bisected because of formely concatenated skips connections
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.op2 = IDAct()

    def forward(self, x):
        out = self.up(x)
        out = self.op2(out)
        return out


