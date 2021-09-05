import torch.nn as nn
from . import basic_blocks as bb


# An Encoder net based on Xception structure
class Xception(nn.Module):
    def __init__(self, in_channels, out_channels=2048, GAP=None, separable_conv=True, cbam=True):
        super(Xception, self).__init__()

        self.entry = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            bb.ResnetBlock(64, 128, 2, separable_conv, cbam, pre_relu=False, max_pool=True),  # 128 X 128 X 128
            bb.ResnetBlock(128, 256, 2, separable_conv, cbam, pre_relu=True, max_pool=True),  # 256 X 64 X 64
            bb.ResnetBlock(256, 728, 2, separable_conv, cbam, pre_relu=True, max_pool=True)   # 728 X 32 X 32
        )

        middle_flow = [bb.ResnetBlock(728, 728, 3, separable_conv, cbam, pre_relu=True, max_pool=False)] * 8
        self.middle = nn.Sequential(*middle_flow)  # 728 X 32 X 32

        self.exit = nn.Sequential(
            bb.ResnetBlock(728, 1024, 2, separable_conv, cbam, pre_relu=True, max_pool=True, diy=[728, 1024]),  # 1024 X 16 X 16
            bb.ResnetBlock(1024, 2048, 2, separable_conv, cbam, max_pool=False, skip=False, diy=[1024, 2048]),  # 2048 X 16 X 16
            nn.Conv2d(2048, out_channels, kernel_size=1)
        )
        # Global Average Pooling
        if GAP is not None:
            self.exit.add_module("2", nn.AdaptiveAvgPool2d(GAP))   # out_channels X GAP

    def forward(self, x):
        x = self.entry(x)
        x = self.middle(x)
        out = self.exit(x)
        return out


class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, separable_conv=True, cbam=False, skip=False):
        super(Decoder, self).__init__()
        self.skip = skip
        self.ups = [bb.ResnetBlock(in_channels, 1024, 2, separable_conv, cbam, pre_relu=False, max_pool=False, skip=False, upsanmple=True)]
        ic = 1024 if skip else 512
        oc = 512
        for i in range(3):
            self.ups.append(bb.ResnetBlock(ic, oc, 2, separable_conv, cbam, pre_relu=True, max_pool=False, skip=False, upsanmple=True))
            oc //= 2
            ic //= 2
        self.exit = nn.Sequential(
            bb.ResnetBlock(ic, oc, 2, separable_conv, cbam, pre_relu=True, max_pool=False, skip=False, upsanmple=True),
            nn.Conv2d(oc//2, out_channels, kernel_size=1)
        )

    def forward(self, x):
        for i in range(4):
            x = self.ups[i](x)
        out = self.exit(x)
        return out