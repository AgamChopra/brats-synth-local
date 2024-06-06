"""
Created on June 2023
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
@Refs:
    - PyTorch 2.0 stable documentation @ https://pytorch.org/docs/stable/
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import pad3d


class attention_grid(nn.Module):
    def __init__(self, x_c, g_c, i_c, stride=2, mode='trilinear'):
        super(attention_grid, self).__init__()
        self.input_filter = nn.Conv3d(
            in_channels=x_c, out_channels=i_c, kernel_size=1,
            stride=stride, bias=False)
        self.gate_filter = nn.Conv3d(
            in_channels=g_c, out_channels=i_c, kernel_size=1,
            stride=1, bias=True)
        self.psi = nn.Conv3d(in_channels=i_c, out_channels=1,
                             kernel_size=1, stride=1, bias=True)
        self.bnorm = nn.InstanceNorm3d(i_c)
        self.mode = mode

    def forward(self, x, g):
        x_shape = x.shape

        a = self.input_filter(x)
        b = self.gate_filter(g)

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
    def __init__(self, in_c, out_c, hid_c=None, final_layer=False,
                 dropout_rate=0.3):
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

            if final_layer:
                self.pool = False
            else:
                self.pool = True
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
                    in_channels=hid_c, out_channels=out_c, kernel_size=2,
                    stride=2),
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


class Attention_UNetT(nn.Module):
    def __init__(self, in_c, out_c, n=1, dropout_rate=0.1):
        super(Attention_UNetT, self).__init__()
        self.out_c = out_c

        self.encoder_layers = nn.ModuleList([
            Block(in_c=in_c, out_c=int(64/n), dropout_rate=dropout_rate),
            Block(in_c=int(64/n), out_c=int(128/n), dropout_rate=dropout_rate),
            Block(in_c=int(128/n), out_c=int(256/n),
                  dropout_rate=dropout_rate),
            Block(in_c=int(256/n), out_c=int(512/n), dropout_rate=dropout_rate)
        ])

        self.latent_layer = Block(in_c=int(512/n), out_c=int(512/n),
                                  hid_c=int(1024/n), dropout_rate=dropout_rate)

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
            attention_grid(int(512/n), int(512/n), int(512/n)),
            attention_grid(int(256/n), int(256/n), int(256/n)),
            attention_grid(int(128/n), int(128/n), int(128/n)),
            attention_grid(int(64/n), int(64/n), int(64/n))
        ])

        self.out = nn.Conv3d(in_channels=int(
            64/n), out_channels=out_c, kernel_size=1)

    def forward(self, x, gans=False):
        target_shape = x.shape[2:]

        y = pad3d(x.float(), 240)  # max(target_shape)

        encoder_outputs = []
        for layer in self.encoder_layers:
            y, y_out = layer(y)
            encoder_outputs.append(y_out)

        y = self.latent_layer(y)

        if gans:
            y += torch.rand_like(y, device=y.device)

        for layer, skip, encoder_output in zip(self.decoder_layers,
                                               self.skip_layers,
                                               encoder_outputs[::-1]):
            y = layer(torch.cat((skip(encoder_output, y)[
                      0], pad3d(y, encoder_output)), dim=1))

        y = self.out(y)

        y = pad3d(y, target_shape)

        return y

    def freeze_encoder(self):
        for layer in self.encoder_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for layer in self.skip_layers:
            for param in layer.parameters():
                param.requires_grad = False
        for param in self.latent_layer.parameters():
            param.requires_grad = False


class Critic(nn.Module):  # (N,3,128,128) -> (N,1)
    def __init__(self, in_c=1, fact=2):
        super(Critic, self).__init__()
        self.c = fact

        self.E = nn.Sequential(nn.utils.spectral_norm(
            nn.Conv3d(in_c, self.c * 2, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=False),
            nn.utils.spectral_norm(
                nn.Conv3d(self.c * 2, self.c * 4, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=False),
            nn.utils.spectral_norm(
                nn.Conv3d(self.c * 4, self.c * 8, kernel_size=5, stride=5)),
            nn.LeakyReLU(0.2, inplace=False)
        )

        self.fc = nn.Sequential(nn.Linear(self.c * 8, 1))

    def forward(self, x):
        target_shape = x.shape[2:]
        y = pad3d(x.float(), max(target_shape))
        y = self.E(y)
        y = self.fc(y.squeeze())
        return y


def test_model(device='cpu', B=1, emb=1, ic=1, oc=1, n=64):
    a = torch.ones((B, ic, 240, 240, 155), device=device)

    model = Attention_UNetT(in_c=ic, out_c=oc, n=n).to(device)
    model.freeze_encoder()
    critic = Critic(in_c=ic, fact=1).to(device)

    b = model(a, gans=True)
    c = critic(b)

    print(a.shape)
    print(b.shape)
    print(c.shape)


if __name__ == '__main__':
    test_model('cpu', B=1, n=64)
