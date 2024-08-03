"""
Created on July 2024
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
from linear_attention import VisionTransformerBlock
from fourier_neural_operator import FourierBlock


class Upsample(nn.Module):
    def __init__(self, in_c=1, out_c=1, dropout=0.3, scale_factor=2,
                 kernel_size=3, stride=1, padding=1):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Sequential(nn.Conv3d(in_c, out_c,
                                            kernel_size=kernel_size,
                                            stride=stride),
                                  nn.InstanceNorm3d(out_c),
                                  nn.GELU(),
                                  nn.Dropout3d(dropout))

    def forward(self, x):
        y = F.interpolate(x, scale_factor=self.scale_factor, mode='nearest')
        y = self.conv(y)
        return y


class DepthwiseFeedForwardBlock(nn.Module):
    def __init__(self, in_c=1, dropout=0.3):
        super(DepthwiseFeedForwardBlock, self).__init__()

        self.passthrough = nn.Sequential(nn.Conv3d(in_c, in_c, kernel_size=1),
                                         nn.Conv3d(in_c, in_c, kernel_size=3, groups=in_c))
        self.gate = nn.Sequential(nn.Conv3d(in_c, in_c, kernel_size=1),
                                  nn.Conv3d(
                                      in_c, in_c, kernel_size=3, groups=in_c),
                                  nn.InstanceNorm3d(in_c),
                                  nn.GELU())
        self.rectifire = nn.Sequential(nn.Conv3d(in_c, in_c, kernel_size=1),
                                       nn.InstanceNorm3d(in_c),
                                       nn.Dropout3d(dropout))

    def forward(self, x):
        y = self.passthrough(x) * self.gate(x)
        y = pad3d(y, x)
        y = self.rectifire(y)
        y = x + y
        return y


class FourierGateAttentionBlock(nn.Module):
    def __init__(self, in_c,
                 img_size, patch_size,
                 embed_dim=512, n_heads=8,
                 mlp_ratio=8, qkv_bias=True,
                 dropout_rate=0.3):
        super(FourierGateAttentionBlock, self).__init__()
        out_c = in_c
        n_classes = img_size ** 3

        self.vision_block = VisionTransformerBlock(
            img_size=img_size,
            patch_size=patch_size,
            in_c=in_c,
            out_c=out_c,
            n_classes=n_classes,
            embed_dim=embed_dim,
            n_heads=n_heads,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            dropout=dropout_rate
        )

        self.frequency_block = FourierBlock(
            in_c, in_c, out_c,
            img_size, dropout_rate)

        self.merge_block = nn.Sequential(nn.Conv3d(2*out_c, out_c,
                                                   kernel_size=1,
                                                   groups=2),
                                         nn.InstanceNorm3d(out_c),
                                         nn.Dropout3d(dropout_rate))

    def forward(self, x):
        y1 = self.frequency_block(x)
        # print('      FreqBlock', y1.mean().item())
        y2 = self.vision_block(x)
        # print('      VisBlock', y2.mean().item())
        y = self.merge_block(torch.cat((y1, y2), dim=1))
        # print('      MergeBlock', y.mean().item())
        y = x + y
        # print('      sum', y.mean().item())
        return y


class TransformerBlock(nn.Module):
    def __init__(self, in_c,
                 img_size, patch_size,
                 embed_dim=512, n_heads=8,
                 mlp_ratio=8, qkv_bias=True,
                 dropout_rate=0.3):
        super(TransformerBlock, self).__init__()

        self.layers = nn.ModuleList([
            FourierGateAttentionBlock(in_c,
                                      img_size, patch_size,
                                      embed_dim, n_heads,
                                      mlp_ratio, qkv_bias,
                                      dropout_rate),
            DepthwiseFeedForwardBlock(in_c, dropout_rate)
        ])

    def forward(self, x):
        y = self.layers[0](x)
        # print('   FGABlock', y.mean().item())
        y = self.layers[1](x)
        # print('   FeedForwardBlock', y.mean().item())
        return y


class TransformerBlockDown(nn.Module):
    def __init__(self, in_c,
                 img_size, patch_size,
                 embed_dim=512, n_heads=8,
                 mlp_ratio=8, qkv_bias=True,
                 dropout_rate=0.3):
        super(TransformerBlockDown, self).__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(in_c,
                             img_size, patch_size,
                             embed_dim, n_heads,
                             mlp_ratio, qkv_bias,
                             dropout_rate),
            nn.Sequential(nn.Conv3d(in_c, 2 * in_c,
                                    kernel_size=3,
                                    stride=2,
                                    padding=1),  # !!! k=2, s=2
                          nn.InstanceNorm3d(2 * in_c),
                          nn.GELU(),
                          nn.Dropout3d(dropout_rate))
        ])

    def forward(self, x):
        y_skip = self.layers[0](x)
        y = self.layers[1](x)
        return y, y_skip


class TransformerBlockUp(nn.Module):
    def __init__(self, in_c,
                 img_size, patch_size,
                 embed_dim=512, n_heads=8,
                 mlp_ratio=8, qkv_bias=True,
                 dropout_rate=0.3, final=False):
        super(TransformerBlockUp, self).__init__()

        self.final = final
        self.layers = nn.Sequential(
            nn.Conv3d(in_c, int(in_c/2), kernel_size=1),
            TransformerBlock(int(in_c/2),
                             img_size, patch_size,
                             embed_dim, n_heads,
                             mlp_ratio, qkv_bias,
                             dropout_rate)
        )
        if not self.final:
            self.up = Upsample(int(in_c/2), int(in_c/4),
                               scale_factor=2, dropout=dropout_rate)

    def forward(self, x):
        y = self.layers(x)
        if not self.final:
            y = self.up(y)
        return y


class TransformerBlockLatant(nn.Module):
    def __init__(self, in_c,
                 img_size, patch_size,
                 embed_dim=512, n_heads=8,
                 mlp_ratio=8, qkv_bias=True,
                 dropout_rate=0.3, mask_downsample=6,
                 noise=True):
        super(TransformerBlockLatant, self).__init__()
        self.noise = noise
        self.noise_mask_layer = nn.Sequential(nn.Conv3d(1, 1,
                                                        kernel_size=mask_downsample,
                                                        stride=mask_downsample),
                                              nn.InstanceNorm3d(1),
                                              nn.Conv3d(1, in_c, kernel_size=1))

        self.layers = nn.Sequential(
            TransformerBlock(in_c,
                             img_size, patch_size,
                             embed_dim, n_heads,
                             mlp_ratio, qkv_bias,
                             dropout_rate),
            Upsample(in_c, int(in_c/2), scale_factor=2,
                     dropout=dropout_rate))

    def forward(self, x, mask):
        encodings = self.noise_mask_layer(mask * (
            torch.rand_like(mask, device=mask.device) if self.noise else 1.))
        # print(x.shape, encodings.shape)
        y = x + encodings
        y = self.layers(y)
        return y


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
            y_out = self.pool_block(y)
            return y_out, y
        else:
            return y
