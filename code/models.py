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
import torch.nn.functional as F

from utils import pad3d
from vision_transformer import VisionTransformer3D


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
            y, y_out = layer(y)
            encoder_outputs.append(y_out)

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
            patch_size=48,
            in_c=self.c * 8,
            n_classes=1,
            embed_dim=32,
            depth=4,
            n_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            dropout=0.
        )

        self.fc2 = VisionTransformer3D(
            img_size=48,
            patch_size=24,
            in_c=self.c * 8,
            n_classes=1,
            embed_dim=32,
            depth=4,
            n_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            dropout=0.
        )

        self.fc3 = VisionTransformer3D(
            img_size=48,
            patch_size=12,
            in_c=self.c * 8,
            n_classes=1,
            embed_dim=32,
            depth=4,
            n_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            dropout=0.
        )

        self.fc4 = VisionTransformer3D(
            img_size=48,
            patch_size=6,
            in_c=self.c * 8,
            n_classes=1,
            embed_dim=32,
            depth=4,
            n_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            dropout=0.
        )

        self.fc5 = VisionTransformer3D(
            img_size=48,
            patch_size=3,
            in_c=self.c * 8,
            n_classes=1,
            embed_dim=32,
            depth=4,
            n_heads=4,
            mlp_ratio=4,
            qkv_bias=False,
            dropout=0.
        )

        self.out = nn.Linear(5, 1)

    def forward(self, x):
        target_shape = x.shape[2:]
        y = pad3d(x.float(), max(target_shape))
        y = self.E(y)
        y1, y2, y3, y4, y5 = self.fc1(y), self.fc2(
            y), self.fc3(y), self.fc4(y), self.fc5(y)
        y = self.out(torch.cat((y1, y2, y3, y4, y5), dim=1))
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
    a = torch.ones((B, ic, 240, 240, 155), device=device)

    model = Attention_UNet(in_c=ic, out_c=oc, n=n).to(device)
    critic1 = Critic(in_c=ic, fact=1).to(device)
    critic2 = Critic_VT(in_c=ic, fact=1).to(device)

    b = model(a)
    c = critic1(b)
    d = critic2(b)

    print(a.shape)
    print(b.shape)
    print(c.shape)
    print(d.shape)


if __name__ == '__main__':
    test_model('cpu', B=1, n=64)
