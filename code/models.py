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

from utils import pad3d, count_parameters
from attention import VisionTransformer3D

from network import AttentionGrid, Block
from network import TransformerBlockDown, TransformerBlockUp, TransformerBlockLatant


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
                 mask_downsample=30,
                 noise=True,
                 device1='cpu', device2='cpu'):
        super(Global_UNet, self).__init__()
        self.out_c = out_c
        self.dropout_rate = dropout_rate
        self.device1 = device1
        self.device2 = device2

        self.downsample = nn.Sequential(
            nn.Conv3d(in_c, in_c * fact, kernel_size=4, stride=4, padding=8),
            nn.InstanceNorm3d(in_c),
            nn.GELU()
        ).to(device=device1)

        self.encoder_layers = nn.ModuleList([
            TransformerBlockDown(in_c * fact, img_size=64, patch_size=16,
                                 dropout_rate=dropout_rate,
                                 embed_dim=embed_dim, qkv_bias=qkv_bias,
                                 mlp_ratio=mlp_ratio),
            TransformerBlockDown(2 * in_c * fact, img_size=32, patch_size=8,
                                 dropout_rate=dropout_rate,
                                 embed_dim=embed_dim, qkv_bias=qkv_bias,
                                 mlp_ratio=mlp_ratio),
            TransformerBlockDown(4 * in_c * fact, img_size=16, patch_size=4,
                                 dropout_rate=dropout_rate,
                                 embed_dim=embed_dim, qkv_bias=qkv_bias,
                                 mlp_ratio=mlp_ratio)
        ]).to(device=device1)

        self.latent_layer = nn.ModuleList([
            TransformerBlockLatant(8 * in_c * fact, img_size=8, patch_size=2,
                                   dropout_rate=dropout_rate,
                                   embed_dim=embed_dim, qkv_bias=qkv_bias,
                                   mlp_ratio=mlp_ratio,
                                   noise=noise,
                                   mask_downsample=mask_downsample)
        ]).to(device=device1)

        self.decoder_layers = nn.ModuleList([
            TransformerBlockUp(8 * in_c * fact, img_size=16, patch_size=4,
                               dropout_rate=dropout_rate,
                               embed_dim=embed_dim, qkv_bias=qkv_bias,
                               mlp_ratio=mlp_ratio),
            TransformerBlockUp(4 * in_c * fact, img_size=32, patch_size=8,
                               dropout_rate=dropout_rate,
                               embed_dim=embed_dim, qkv_bias=qkv_bias,
                               mlp_ratio=mlp_ratio),
            TransformerBlockUp(2 * in_c * fact, img_size=64, patch_size=16,
                               dropout_rate=dropout_rate,
                               embed_dim=embed_dim, qkv_bias=qkv_bias,
                               mlp_ratio=mlp_ratio, final=True)
        ]).to(device=device2)

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_c * fact, in_c,
                               kernel_size=4, stride=4, padding=8),
            nn.InstanceNorm3d(in_c),
            nn.GELU(),
            nn.Conv3d(in_c, out_c, kernel_size=1)
        ).to(device=device2)

    def forward(self, x, mask):
        target_shape = x.shape[2:]

        y = pad3d(x.float(), 240)
        latent_mask = pad3d(mask.float(), 240)

        y = y.to(device=self.device1)
        y = self.downsample(y)

        encoder_outputs = []
        for layer in self.encoder_layers:
            y, y_skip = layer(y)
            encoder_outputs.append(y_skip.to(self.device2))

        for layer in self.latent_layer:
            #print(y.shape, mask.shape)
            y = layer(y, latent_mask)

        y = y.to(device=self.device2)
        for layer, encoder_output in zip(self.decoder_layers,
                                         encoder_outputs[::-1]):
            # print(y.shape, encoder_output.shape)
            y = layer(
                torch.cat((pad3d(y, encoder_output), encoder_output), dim=1))

        y = self.upsample(y)
        y = pad3d(y, target_shape)
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


class DepthwiseSeparableConv3d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(DepthwiseSeparableConv3d, self).__init__()
        self.depthwise = nn.utils.spectral_norm(
            nn.Conv3d(in_channels, in_channels, kernel_size=kernel_size,
                      stride=stride, groups=in_channels))
        self.pointwise = nn.utils.spectral_norm(
            nn.Conv3d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class CriticA(nn.Module):
    """
    Critic model for adversarial training.

    Args:
        in_c (int, optional): Number of input channels. Default is 1.
        fact (int, optional): Scaling factor for the number of channels. Default is 1.
    """

    def __init__(self, in_c=1, fact=8):
        super(CriticA, self).__init__()
        self.c = fact

        self.E = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(in_c, self.c * 2, kernel_size=3, stride=3)),
            nn.GELU(),
            DepthwiseSeparableConv3d(
                self.c * 2, self.c * 4, kernel_size=2, stride=2),
            nn.GELU(),
            DepthwiseSeparableConv3d(
                self.c * 4, self.c * 8, kernel_size=3, stride=3),
            nn.GELU(),
            DepthwiseSeparableConv3d(
                self.c * 8, self.c * 16, kernel_size=2, stride=2),
            nn.GELU(),
            DepthwiseSeparableConv3d(
                self.c * 16, self.c * 32, kernel_size=3, stride=3),
            nn.GELU(),
            DepthwiseSeparableConv3d(
                self.c * 32, self.c * 64, kernel_size=2, stride=2),
            nn.GELU(),
            nn.utils.spectral_norm(
                nn.Conv3d(self.c * 64, 1, kernel_size=1, stride=1))
        )

    def forward(self, x):
        y = pad3d(x.float(), 240)
        y = self.E(y).squeeze()
        return y


class Critic(nn.Module):
    """
    Critic model for adversarial training.

    Args:
        in_c (int, optional): Number of input channels. Default is 1.
        fact (int, optional): Scaling factor for the number of channels. Default is 1.
    """

    def __init__(self, in_c=1, fact=1):
        super(Critic, self).__init__()
        self.c = fact

        self.E = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(in_c, self.c * 8, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv3d(self.c * 8, self.c * 8, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=True),
            nn.utils.spectral_norm(
                nn.Conv3d(self.c * 8, self.c * 8, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.fc = nn.Sequential(nn.Linear(self.c * 8, 1))

    def forward(self, x):
        target_shape = x.shape[2:]
        y = pad3d(x.float(), max(target_shape))
        y = self.E(y)
        y = self.fc(y.squeeze())
        return y


class Critic_VT(nn.Module):
    """
    Vision Transformer-based Critic model for adversarial training.

    Args:
        in_c (int, optional): Number of input channels. Default is 1.
        fact (int, optional): Scaling factor for the number of channels. Default is 1.
    """

    def __init__(self, in_c=1, fact=1):
        super(Critic_VT, self).__init__()
        self.c = fact

        self.E = nn.Sequential(
            nn.utils.spectral_norm(
                nn.Conv3d(in_c, self.c * 8, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.fc1 = VisionTransformer3D(
            img_size=48,
            patch_size=12,
            in_c=self.c * 8,
            n_classes=1,
            embed_dim=64,
            depth=4,
            n_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            dropout=0.
        )

        self.fc2 = VisionTransformer3D(
            img_size=48,
            patch_size=4,
            in_c=self.c * 8,
            n_classes=1,
            embed_dim=64,
            depth=4,
            n_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            dropout=0.
        )

        self.out = nn.Linear(2, 1)

    def forward(self, x):
        target_shape = x.shape[2:]
        y = pad3d(x.float(), max(target_shape))
        y = self.E(y)
        y1, y2 = self.fc1(y), self.fc2(y)
        y = self.out(torch.cat((y1, y2), dim=1))
        return y.view(x.shape[0])


def test_model(device='cpu', B=1, emb=1, ic=1, oc=1, n=64):
    """
    Test function to instantiate and test the models.

    Args:
        device (str, optional): Device to run the models on. Default is 'cpu'.
        B (int, optional): Batch size. Default is 1.
        emb (int, optional): Embedding dimension. Default is 1.
        ic (int, optional): Input channels. Default is 1.
        oc (int, optional): Output channels. Default is 1.
        n (int, optional): Scaling factor for channels. Default is 64.
    """
    a = torch.ones((B, ic, 240, 240, 240), device=device)
    mask = torch.ones((B, 1, 240, 240, 240), device=device)

    model = Global_UNet(in_c=ic, out_c=oc, fact=4,
                        embed_dim=32, n_heads=4,
                        mlp_ratio=4, qkv_bias=True,
                        dropout_rate=0.,
                        mask_downsample=30,
                        noise=True, device1=device, device2=device)
    print(f'Generator size: {int(count_parameters(model)/1000000)}M')

    critic = CriticA(in_c=2, fact=2).to(device)
    print(f'Critic size: {int(count_parameters(critic)/1000000)}M')

    optimizer = torch.optim.AdamW(
        list(model.parameters()) + list(critic.parameters()), lr=1e-4)

    b = model(a, mask)
    c = critic(torch.cat((b, a), dim=1))

    print(a.shape, mask.shape)
    print(b.shape)
    print(c, c.shape)

    error = -c
    error.backward()
    optimizer.step()
    optimizer.zero_grad()


if __name__ == '__main__':
    test_model('cuda', B=1, n=64)
