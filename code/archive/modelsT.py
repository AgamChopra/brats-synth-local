"""
Created on June 2024
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pad3d


class TransformerBlock(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super(TransformerBlock, self).__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, batch_first=True),
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, mlp_dim),
                    nn.GELU(),
                    nn.Linear(mlp_dim, dim)
                )
            ]))

    def forward(self, x):
        for norm1, attn, mlp in self.layers:
            x = attn(norm1(x), x, x)[0] + x
            x = mlp(x) + x
        return x


class TransformerFFT3D(nn.Module):
    def __init__(self, num_classes, dim, depth, heads, mlp_dim):
        super(TransformerFFT3D, self).__init__()

        self.transformer = TransformerBlock(dim, depth, heads, mlp_dim)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes),
        )

    def forward(self, x):
        x = self.transformer(x.permute(0, 2, 1))

        x = self.mlp_head(x).permute(0, 2, 1)
        return x


class Transformer_Model(nn.Module):
    def __init__(self, depth=16):
        super(Transformer_Model, self).__init__()
        self.layer = nn.ModuleList([
            TransformerFFT3D(num_classes=2*32, dim=32*4, depth=depth,
                             heads=16, mlp_dim=1024)
        ])

    def forward(self, x):
        y = x.view(x.shape[0], 4*32, 33*32)

        for layer in self.layer:
            y = layer(y)

        return y.view(x.shape[0], 2, 32, 32, 33)


class AttentionGrid(nn.Module):
    """
    Attention Grid module for 3D convolutions.

    Args:
        x_c (int): Number of channels in the input tensor.
        g_c (int): Number of channels in the gating signal.
        i_c (int): Number of channels for the intermediate computations.
        stride (int, optional): Stride for the input filter convolution. Default is 2.
        mode (str, optional): Mode for interpolation. Default is 'trilinear'.
    """

    def __init__(self, x_c, g_c, i_c, stride=2, mode='trilinear'):
        super(AttentionGrid, self).__init__()
        self.input_filter = nn.Conv3d(
            in_channels=x_c, out_channels=i_c, kernel_size=1, stride=stride, bias=False)
        self.gate_filter = nn.Conv3d(
            in_channels=g_c, out_channels=i_c, kernel_size=1, stride=1, bias=True)
        self.psi = nn.Conv3d(in_channels=i_c, out_channels=1,
                             kernel_size=1, stride=1, bias=True)
        self.bnorm = nn.InstanceNorm3d(i_c)
        self.mode = mode

    def forward(self, x, g):
        x_shape = x.shape
        a = self.input_filter(x)
        b = self.gate_filter(g)

        # Padding to match shapes
        if a.shape[-1] < b.shape[-1]:
            a = pad3d(a, b)
        elif a.shape[-1] > b.shape[-1]:
            b = pad3d(b, a)

        w = torch.sigmoid(self.psi(F.relu(a + b)))
        w = F.interpolate(w, size=x_shape[2:], mode=self.mode)

        y = x * w
        y = self.bnorm(y)
        return y, w


class Block(nn.Module):
    """
    Block module for the U-Net architecture with optional dropout and InstanceNorm3D.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        hid_c (int, optional): Number of hidden channels. If not provided, uses a simpler block.
        final_layer (bool, optional): If true, disables pooling in the block. Default is False.
        dropout_rate (float, optional): Dropout rate. Default is 0.3.
    """

    def __init__(self, in_c, out_c, hid_c=None, final_layer=False, dropout_rate=0.3):
        super(Block, self).__init__()
        if hid_c is None:
            self.layer = nn.Sequential(
                nn.Conv3d(in_channels=in_c, out_channels=out_c,
                          kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(out_c)
            )

            self.out_block = nn.Sequential(
                nn.Conv3d(in_channels=out_c, out_channels=out_c,
                          kernel_size=2, padding=0),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(out_c)
            )

            self.pool = not final_layer
            if self.pool:
                self.pool_block = nn.Sequential(
                    nn.Conv3d(in_channels=out_c, out_channels=out_c,
                              kernel_size=2, stride=2),
                    nn.ReLU(),
                    nn.Dropout3d(dropout_rate),
                    nn.InstanceNorm3d(out_c)
                )

        else:
            self.pool = False

            self.layer = nn.Sequential(
                nn.Conv3d(in_channels=in_c, out_channels=hid_c,
                          kernel_size=3, padding=0),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(hid_c)
            )

            self.out_block = nn.Sequential(
                nn.Conv3d(in_channels=hid_c, out_channels=hid_c,
                          kernel_size=2, padding=0),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(hid_c),
                nn.ConvTranspose3d(
                    in_channels=hid_c, out_channels=out_c, kernel_size=2, stride=2),
                nn.ReLU(),
                nn.Dropout3d(dropout_rate),
                nn.InstanceNorm3d(out_c)
            )

    def forward(self, x):
        y = self.layer(x)
        y = self.out_block(y)

        if self.pool:
            y_ = self.pool_block(y)
            return y_, y
        else:
            return y


class Attention_UNet(nn.Module):
    """
    Attention U-Net with optional Vision Transformer in the latent layer.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        n (int, optional): Scale factor for the number of channels. Default is 1.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
        vision_transformer (bool, optional): Whether to use Vision Transformer in the latent layer. Default is False.
    """

    def __init__(self, in_c, out_c, n=1, dropout_rate=0.1):
        super(Attention_UNet, self).__init__()
        self.out_c = out_c

        self.encoder_layers = nn.ModuleList([
            Block(in_c=in_c, out_c=int(64/n), dropout_rate=dropout_rate),
            Block(in_c=int(64/n), out_c=int(128/n), dropout_rate=dropout_rate),
            Block(in_c=int(128/n), out_c=int(256/n),
                  dropout_rate=dropout_rate),
            Block(in_c=int(256/n), out_c=int(512/n), dropout_rate=dropout_rate)
        ])

        self.latent_layers = nn.ModuleList([
            Block(in_c=int(512/n), out_c=int(512/n),
                  hid_c=int(1024/n), dropout_rate=dropout_rate)
        ])

        self.decoder_layers = nn.ModuleList([
            Block(in_c=int(1024/n), out_c=int(256/n),
                  hid_c=int(512/n), dropout_rate=dropout_rate),
            Block(in_c=int(512/n), out_c=int(128/n),
                  hid_c=int(256/n), dropout_rate=dropout_rate),
            Block(in_c=int(256/n), out_c=int(64/n),
                  hid_c=int(128/n), dropout_rate=dropout_rate),
            Block(in_c=int(128/n), out_c=int(64/n),
                  final_layer=True, dropout_rate=dropout_rate)
        ])

        self.skip_layers = nn.ModuleList([
            AttentionGrid(int(512/n), int(512/n), int(512/n)),
            AttentionGrid(int(256/n), int(256/n), int(256/n)),
            AttentionGrid(int(128/n), int(128/n), int(128/n)),
            AttentionGrid(int(64/n), int(64/n), int(64/n))
        ])

        self.out = nn.Conv3d(in_channels=int(
            64/n), out_channels=out_c, kernel_size=1)

    def forward(self, x, gans=False):
        target_shape = x.shape[2:]

        y = pad3d(x.float(), 240)

        encoder_outputs = []
        for layer in self.encoder_layers:
            y, y_out = layer(y)
            if gans:
                y_out = nn.functional.dropout3d(y_out, p=0.3)
                y = nn.functional.dropout3d(y, p=0.3)
            encoder_outputs.append(y_out)

        for layer in self.latent_layers:
            y = layer(y)
            if gans:
                y = nn.functional.dropout3d(y, p=0.3)

        for layer, skip, encoder_output in zip(self.decoder_layers, self.skip_layers, encoder_outputs[::-1]):
            y = layer(torch.cat((skip(encoder_output, y)[
                      0], pad3d(y, encoder_output)), dim=1))
            if gans:
                y = nn.functional.dropout3d(y, p=0.3)

        y = self.out(y)
        y = pad3d(y, target_shape)
        return y


def test_model(device='cpu', B=1, emb=1):
    from utils import get_fft_mag_phase, reconst_image, norm
    from dataloader import show_images
    with torch.no_grad():
        a0 = torch.rand((B, 1, 240, 240, 240), device=device)
        mask = torch.rand((B, 1, 240, 240, 240), device=device)
        a = torch.cat((get_fft_mag_phase(a0), get_fft_mag_phase(mask)), dim=1)
        print(a0.shape, mask.shape)
        print(a.shape)

        model = Transformer_Model().to(device)
        model_style_transfer = Attention_UNet(2, 1, n=32).to(device)

        b = model(a)
        c = model_style_transfer(torch.cat((reconst_image(b), a0), dim=1))
        print(b.shape)
        print(c.shape)

        show_images(
            torch.cat(
                (norm(a0).cpu(), norm(reconst_image(b)), c), dim=0), 3, 3, dpi=350)


if __name__ == '__main__':
    test_model('cpu', B=1)
