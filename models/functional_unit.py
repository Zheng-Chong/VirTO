import torch
import torch.nn as nn
from . import basic_blocks as bb


# An Encoder net based on U-Net structure
class UNetEncoder(nn.Module):
    def __init__(self, in_channels, cbam=False):
        super(UNetEncoder, self).__init__()
        self.down_blocks = [bb.UNetBlock(in_channels, 64)]  # 64 X 512 X 512
        self.down_blocks += [bb.UNetBlock(64, 128, conv=bb.SeparableConv, max_pool=True)]  # 128 X 256 X 256
        self.down_blocks += [bb.UNetBlock(128, 256, conv=bb.SeparableConv, max_pool=True)]  # 256 X 128 X 128
        self.down_blocks += [bb.UNetBlock(256, 512, conv=bb.SeparableConv, max_pool=True)]  # 512 X 64 X 64
        self.exit = bb.UNetBlock(512, 1024, conv=bb.SeparableConv, max_pool=True, upsample=True)  # 512 X 64 X 64

    def forward(self, x):
        skip_res = []
        middle_res = x
        for block in self.down_blocks:
            middle_res = block(middle_res)
            skip_res.append(middle_res)
        return self.exit(middle_res), skip_res


# An Encoder net based on Xception structure
class Xception(nn.Module):
    def __init__(self, in_channels, GAP=None, cbam=False):
        super(Xception, self).__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),  # 32 X 256 X 256
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # 64 X 256 X 256
            nn.BatchNorm2d(64),
            nn.ReLU(),
            bb.XceptionBlock(64, 128, 2, cbam=cbam, max_pool=True, skip=True),  # 128 X 128 X 128
            bb.XceptionBlock(128, 256, 2, cbam=cbam, pre_relu=True, max_pool=True, skip=True),  # 256 X 64 X 64
            bb.XceptionBlock(256, 728, 2, cbam=cbam, pre_relu=True, max_pool=True, skip=True)   # 728 X 32 X 32
        )

        middle_flow = [bb.XceptionBlock(728, 728, 3, cbam=cbam, pre_relu=True, skip=True)] * 8
        self.middle = nn.Sequential(*middle_flow)  # 728 X 32 X 32

        self.exit = nn.Sequential(
            bb.XceptionBlock(728, 1024, 2, cbam=cbam, pre_relu=True, max_pool=True, diy=[728, 1024]),  # 1024 X 16 X 16
            bb.XceptionBlock(1024, 2048, 2, cbam=cbam, max_pool=False, diy=[1536, 2048]),  # 2048 X 16 X 16
        )
        # Global Average Pooling
        if GAP is not None:
            self.exit.add_module("2", nn.AdaptiveAvgPool2d(GAP))   # out_channels X GAP

    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        out = self.exit(x)
        return out.view(out.size(0), -1, 32, 32)


# An Decoder net based on U-Net structure
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, conv=nn.Conv2d, cbam=False):
        super(UNetDecoder, self).__init__()
        self.ups = [bb.UNetBlock(in_channels, 512, conv, cbam=cbam, upsample=True),
                    bb.UNetBlock(512, 256, conv, cbam=cbam, upsample=True),
                    bb.UNetBlock(256, 128, conv, cbam=cbam, upsample=True),
                    bb.UNetBlock(128, 64, conv, cbam=cbam)
                    ]
        self.exit = nn.Sequential(nn.Conv2d(64, out_channels, kernel_size=1),
                                  nn.ReLU()
                                  )

    def forward(self, x, skip_x):
        middle_res = x
        for i in range(4):
            middle_res = torch.cat((middle_res, skip_x[3-i]), 1)
            middle_res = self.ups[i](middle_res)
        return self.exit(middle_res)
