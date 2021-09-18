from . import functional_unit as fu
from . import basic_blocks as bb
import torch.nn as nn
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Cloth Mask Generator
class CMGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, cbam=False):
        super(CMGenerator, self).__init__()
        self.enc = fu.UNetEncoder(in_channels, cbam=cbam)
        self.dec = fu.UNetDecoder(512*2, out_channels, conv=bb.SeparableConv, cbam=cbam)
        self.relu = nn.ReLU()

    def forward(self, cloth_img):
        i, skip = self.enc(cloth_img)
        return self.dec(i, skip)


# Generator of PGM
class PGMGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, cbam=False):
        super(PGMGenerator, self).__init__()
        self.cloth_enc = fu.Xception(in_channels, cbam=cbam)
        self.iuv_enc = fu.UNetEncoder(in_channels, cbam=cbam)
        
        self.dec = fu.UNetDecoder(512 * 2 + 128, out_channels, conv=bb.SeparableConv, cbam=cbam)

    def forward(self, iuv, cloth_mask):
        c = self.cloth_enc(cloth_mask)
        i, skip = self.iuv_enc(iuv)
        code = torch.cat((i, c), 1)
        return self.dec(code, skip)


# Discriminator Network
# 3 down-sampling to extract features -> fc layer to judge true and false
class Discriminator(nn.Module):
    def __init__(self, in_channels):
        super(Discriminator, self).__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(in_channels, 16 * in_channels, 3, padding=1, groups=in_channels),  # 16 X 32 X 32
            nn.MaxPool2d(3, 2, padding=1),   # 16 X 16 X 16
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16 * in_channels, 64 * in_channels, 3, padding=1),  # 64 X 16 X 16
            nn.MaxPool2d(3, 2, padding=1),   # 64 X 8 X 8
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64 * in_channels, 128 * in_channels, 3, padding=1),  # 128 X 8 X 8
            nn.AdaptiveMaxPool2d((1, 1)),  # 128 X 1 X 1
            nn.Conv2d(128 * in_channels, in_channels, 1, groups=in_channels)  # 1 X 1 X 1
        )

    def forward(self, x):
        f = self.enc(x)
        return f.view(f.size(0), f.size(1), -1)


# Adversarial Loss based on PatchGAN
class AdversarialLoss(nn.Module):
    def __init__(self, lsgan=False):
        super(AdversarialLoss, self).__init__()
        self.loss = nn.MSELoss() if lsgan else nn.BCELoss()

    def forward(self, pred, discriminator, patch_size, is_adv, target=None):
        loss = 0
        valid = torch.ones(pred.size(0), pred.size(1), 1).to(device)
        zeros = torch.zeros(pred.size(0), pred.size(1), 1).to(device)
        row = int(pred.size(2) / patch_size)
        col = int(pred.size(3) / patch_size)
        for a in range(row):
            x = patch_size * a
            if x == row - 1:
                x = pred.size(2) - patch_size
            for b in range(col):
                y = patch_size * b
                if y == col - 1:
                    y = pred.size(3) - patch_size
                patch_fake = pred[:, :, x:x + patch_size, y:y + patch_size]
                pred_fake = discriminator(patch_fake)
                if is_adv:
                    loss += (self.loss(pred_fake, valid))
                    continue
                patch_real = target[:, :, x:x + patch_size, y:y + patch_size]
                pred_real = discriminator(patch_real)
                loss += torch.div(self.loss(pred_real, valid), 2) + torch.div(self.loss(pred_fake, zeros), 2)
        return loss / (row * col)
