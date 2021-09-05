from . import functional_unit as fu
import torch.nn as nn
import torch


# Generator of PGM
class PGMGenerator(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, code_channels=2048):
        super(PGMGenerator, self).__init__()
        self.shape_enc = fu.Xception(in_channels, code_channels)
        self.dec = fu.Decoder(code_channels*2, out_channels)

    def forward(self, iuv, cloth_mask):
        code = torch.cat((self.shape_enc(iuv), self.shape_enc(cloth_mask)), 1)
        return self.dec(code)
