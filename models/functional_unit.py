import torch.nn as nn
import torch
from . import basic_blocks as bb


class ShapeEncoder(nn.Module):
    def __init__(self, n_stages, nf=128, nf_out=3, n_rnb=2,
                 conv_layer=bb.NormConv2d, spatial_size=256, final_act=True, dropout_prob=0.0,):
        super(ShapeEncoder, self).__init__()
        assert (2 ** (n_stages - 1)) == spatial_size
        self.final_act = final_act
        self.blocks = nn.ModuleDict()
        self.ups = nn.ModuleDict()
        self.n_stages = n_stages
        self.n_rnb = n_rnb
        for i_s in range(self.n_stages - 2, 0, -1):
            # for final stage, bisect number of filters
            if i_s == 1:
                # upsampling operations
                self.ups.update({f"s{i_s+1}": bb.Upsample(in_channels=nf, out_channels=nf // 2, conv_layer=conv_layer,)})
                nf = nf // 2
            else:
                # upsampling operations
                self.ups.update({f"s{i_s+1}": bb.Upsample(in_channels=nf, out_channels=nf, conv_layer=conv_layer,)})

            # resnet blocks
            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                self.blocks.update(
                    {
                        stage: VUnetResnetBlock(
                            nf,
                            use_skip=True,
                            conv_layer=conv_layer,
                            dropout_prob=dropout_prob,
                        )
                    }
                )

        # final 1x1 convolution
        self.final_layer = conv_layer(nf, nf_out, kernel_size=1)

        # conditionally: set final activation
        if self.final_act:
            self.final_act = nn.Tanh()

    def forward(self, x, skips):
        """
        Parameters
        ----------
        x : torch.Tensor
            Latent representation to decode.
        skips : dict
            The skip connections of the VUnet
        Returns
        -------
        out : torch.Tensor
            An image as described by :attr:`x` and :attr:`skips`
        """
        out = x
        for i_s in range(self.n_stages - 2, 0, -1):
            out = self.ups[f"s{i_s+1}"](out)

            for ir in range(self.n_rnb, 0, -1):
                stage = f"s{i_s}_{ir}"
                out = self.blocks[stage](out, skips[stage])

        out = self.final_layer(out)
        if self.final_act:
            out = self.final_act(out)
        return out