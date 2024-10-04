"""
Created on June 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
"""

import torch
import torch.nn as nn

from math import ceil

from network import AttentionGrid, Block, Upsample
from network import TransformerBlockDown, TransformerBlockUp, TransformerBlockLatant


def pad3d(inpt, target):
    """
    Pad or crop input image to match target size.

    Args:
        inpt (torch.tensor): Input tensor to be padded or cropped of shape (B, C, X, Y, Z).
        target (torch.tensor or tuple): Target tensor of shape (B, C, X, Y, Z) or tuple of shape (X, Y, Z).

    Returns:
        torch.tensor: Resized (padded or cropped) input tensor matching size of target.
    """
    if torch.is_tensor(target):
        delta = [target.shape[2+i] - inpt.shape[2+i] for i in range(3)]
    else:
        try:
            delta = [target[i] - inpt.shape[2+i] for i in range(3)]
        except Exception:
            delta = [target - inpt.shape[2+i] for i in range(3)]

    return nn.functional.pad(
        input=inpt,
        pad=(
            ceil(delta[2]/2), delta[2] - ceil(delta[2]/2),
            ceil(delta[1]/2), delta[1] - ceil(delta[1]/2),
            ceil(delta[0]/2), delta[0] - ceil(delta[0]/2)
        ),
        mode='constant', value=0.0
    ).to(dtype=inpt.dtype, device=inpt.device)


class Global_UNet(nn.Module):
    """
    Gated Linear Transformer based Global Attention U-Net with fourier neural blocks.

    Args:
        in_c (int): Number of input channels.
        out_c (int): Number of output channels.
        dropout_rate (float, optional): Dropout rate. Default is 0.1.
    """

    def __init__(self, in_c, out_c,
                 fact=32,
                 embed_dim=512, n_heads=8,
                 mlp_ratio=8, qkv_bias=True,
                 dropout_rate=0.3,
                 mask_downsample=16,
                 noise=True,
                 device1='cpu', device2='cpu'):
        super(Global_UNet, self).__init__()
        self.out_c = out_c
        self.dropout_rate = dropout_rate
        self.device1 = device1
        self.device2 = device2

        self.downsample = nn.Sequential(
            nn.Conv3d(in_c, in_c * fact, kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm3d(in_c * fact),
            nn.GELU()
        ).to(device=device1)

        self.encoder_layers = nn.ModuleList([
            TransformerBlockDown(in_c * fact, img_size=120, patch_size=24,
                                 dropout_rate=dropout_rate,
                                 embed_dim=embed_dim, qkv_bias=qkv_bias,
                                 mlp_ratio=mlp_ratio),
            TransformerBlockDown(2 * in_c * fact, img_size=60, patch_size=12,
                                 dropout_rate=dropout_rate,
                                 embed_dim=embed_dim, qkv_bias=qkv_bias,
                                 mlp_ratio=mlp_ratio),
            TransformerBlockDown(4 * in_c * fact, img_size=30, patch_size=6,
                                 dropout_rate=dropout_rate,
                                 embed_dim=embed_dim, qkv_bias=qkv_bias,
                                 mlp_ratio=mlp_ratio)
        ]).to(device=device1)

        self.latent_layer = nn.ModuleList([
            TransformerBlockLatant(8 * in_c * fact, img_size=15, patch_size=5,
                                   dropout_rate=dropout_rate,
                                   embed_dim=embed_dim, qkv_bias=qkv_bias,
                                   mlp_ratio=mlp_ratio,
                                   noise=noise,
                                   mask_downsample=mask_downsample)
        ]).to(device=device1)

        self.decoder_layers = nn.ModuleList([
            TransformerBlockUp(8 * in_c * fact, img_size=30, patch_size=6,
                               dropout_rate=dropout_rate,
                               embed_dim=embed_dim, qkv_bias=qkv_bias,
                               mlp_ratio=mlp_ratio),
            TransformerBlockUp(4 * in_c * fact, img_size=60, patch_size=12,
                               dropout_rate=dropout_rate,
                               embed_dim=embed_dim, qkv_bias=qkv_bias,
                               mlp_ratio=mlp_ratio),
            TransformerBlockUp(2 * in_c * fact, img_size=120, patch_size=24,
                               dropout_rate=dropout_rate,
                               embed_dim=embed_dim, qkv_bias=qkv_bias,
                               mlp_ratio=mlp_ratio, final=True)
        ]).to(device=device2)

        self.upsample = nn.Sequential(
            nn.Conv3d(in_c * fact, out_c,
                      kernel_size=7,
                      stride=1,
                      padding=3),
            nn.InstanceNorm3d(in_c),
            nn.GELU(),
            nn.Dropout3d(dropout_rate),
            nn.Conv3d(out_c, out_c, kernel_size=1)
        ).to(device=device2)

    def forward(self, x, mask):
        target_shape = x.shape[2:]

        y = pad3d(x.float(), 240)
        latent_mask = pad3d(mask.float(), 240)

        y = y.to(device=self.device1)
        y = self.downsample(y)
        y_structure = y
        # print('\n...model...')
        # print('downsample', y.mean().item())

        encoder_outputs = []
        for layer in self.encoder_layers:
            y, y_skip = layer(y)
            encoder_outputs.append(y_skip.to(self.device2))
            # print('encoder', y.shape, y_skip.shape)
            # print('encoder', y.mean().item(), y_skip.mean().item())

        for layer in self.latent_layer:
            # print(y.shape, mask.shape)
            y = layer(y, latent_mask)
            # print('latent', y.mean().item())

        y = y.to(device=self.device2)
        for layer, encoder_output in zip(self.decoder_layers,
                                         encoder_outputs[::-1]):
            # print(y.shape, encoder_output.shape)
            y = layer(
                torch.cat((pad3d(y, encoder_output), encoder_output), dim=1))
            # print('decoder', y.mean().item())

        # print(y.shape)
        y = y + y_structure
        y = self.upsample(y)
        # print('upsample', y.mean().item())
        # print(y.shape)
        # y = nn.functional.sigmoid(y)
        y = nn.functional.interpolate(y, size=240, mode='trilinear')
        y = pad3d(y, target=target_shape)
        # y = y * nn.functional.sigmoid(x + mask)
        # print('........\n')
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
        self.dropout_rate = dropout_rate

        self.encoder_layers = nn.ModuleList([
            Block(in_c=in_c, out_c=int(64/n), dropout_rate=dropout_rate),
            Block(in_c=int(64/n), out_c=int(128/n), dropout_rate=dropout_rate),
            Block(in_c=int(128/n), out_c=int(256/n),
                  dropout_rate=dropout_rate),
            Block(in_c=int(256/n), out_c=int(512/n), dropout_rate=dropout_rate)
        ])

        self.latent_layer = nn.ModuleList([
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

    def forward(self, x):
        target_shape = x.shape[2:]

        y = pad3d(x.float(), 240)

        encoder_outputs = []
        for layer in self.encoder_layers:
            y, y_skip = layer(y)
            encoder_outputs.append(y_skip)

        for layer in self.latent_layer:
            y = (0.8 - self.dropout_rate) * y + (0.2 + self.dropout_rate) * \
                torch.rand_like(y, device=y.device)
            y = layer(y)

        for layer, skip, encoder_output in zip(self.decoder_layers,
                                               self.skip_layers,
                                               encoder_outputs[::-1]):
            y = layer(torch.cat((skip(encoder_output, y)[
                      0], pad3d(y, encoder_output)), dim=1))

        y = self.out(y)
        y = pad3d(y, target_shape)
        return y


class CriticA(nn.Module):
    def __init__(self, in_c, fact=1):
        super(CriticA, self).__init__()
        self.conv1 = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=in_c, out_channels=16*fact, kernel_size=3, stride=2, padding=1))
        self.conv2 = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=16*fact, out_channels=32*fact, kernel_size=3, stride=2, padding=1))
        self.conv3 = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=32*fact, out_channels=64*fact, kernel_size=3, stride=2, padding=1))
        self.conv4 = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=64*fact, out_channels=128*fact, kernel_size=3, stride=2, padding=1))
        self.conv5 = nn.utils.spectral_norm(nn.Conv3d(
            in_channels=128*fact, out_channels=256*fact, kernel_size=3, stride=2, padding=1))

        self.flatten = nn.Flatten()
        self.fc1 = nn.utils.spectral_norm(
            nn.Linear(256 * 8 * 8 * 8 * fact, 512 * fact))
        self.fc2 = nn.utils.spectral_norm(nn.Linear(512*fact, 1))

    def forward(self, x_in):
        x = pad3d(x_in, target=240)
        x = nn.LeakyReLU(0.2)(self.conv1(x))
        x = nn.LeakyReLU(0.2)(self.conv2(x))
        x = nn.LeakyReLU(0.2)(self.conv3(x))
        x = nn.LeakyReLU(0.2)(self.conv4(x))
        x = nn.LeakyReLU(0.2)(self.conv5(x))

        x = self.flatten(x)
        x = nn.LeakyReLU(0.2)(self.fc1(x))
        x = self.fc2(x)

        return x.squeeze()

