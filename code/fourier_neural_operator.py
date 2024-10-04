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

from math import ceil


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


class ElementWiseLayer(nn.Module):
    def __init__(self, c, n):
        super(ElementWiseLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(1, c, n, n, int(n/2) + 1))
        self.B = nn.Parameter(torch.empty(1, c, n, n, int(n/2) + 1))
        nn.init.zeros_(self.B)
        # Xavier initialization for W
        nn.init.xavier_normal_(self.W)
        # He initialization for W (if ReLU or similar activation is used)
        # nn.init.kaiming_normal_(self.W, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        return x * self.W + self.B


def get_fft(x):
    fft_image = torch.fft.rfftn(x)
    fft_shifted = torch.fft.fftshift(fft_image)
    real = fft_shifted.real
    imag = fft_shifted.imag

    real_ = [real.min(), real.max()]
    imag_ = [imag.min(), imag.max()]
    real = (real - real_[0]) / (real_[1] - real_[0])
    imag = (imag - imag_[0]) / (imag_[1] - imag_[0])
    return torch.cat((real, imag), dim=1), real_, imag_


def get_filtered_inverse_fft(freq, shape):
    real_imag = pad3d(freq, (shape, shape, int(shape/2) + 1))
    real, imag = real_imag[:, :int(
        freq.shape[1]/2)], real_imag[:, int(freq.shape[1]/2):]

    fft_reconstructed = torch.complex(real, imag)
    ifft_shifted = torch.fft.ifftshift(fft_reconstructed)
    reconstructed_image = torch.fft.irfftn(
        ifft_shifted, s=(shape, shape, shape)).real

    return reconstructed_image


class FourierSpectralBlock(nn.Module):
    def __init__(self, in_features, out_features, img_size, dropout=0.,
                 crop_ratio=0.25):
        super(FourierSpectralBlock, self).__init__()
        self.spatial_layer = nn.Conv3d(
            in_features, out_features, kernel_size=1, stride=1,
            groups=in_features)

        self.img_size = img_size
        self.cropped = int(img_size * crop_ratio)

        self.frequency_layer = nn.Sequential(
            nn.InstanceNorm3d(in_features*2),
            ElementWiseLayer(in_features*2, self.cropped),
            nn.Conv3d(in_features*2, out_features*2,
                      kernel_size=1, stride=1,
                      groups=in_features*2)
        )

        self.non_linearity = nn.GELU()
        self.norm = nn.InstanceNorm3d(out_features)
        self.drop = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Conv3d(
                in_features*2, out_features, kernel_size=1, stride=1),
            nn.InstanceNorm3d(out_features),
            nn.Sigmoid()
        )

    def forward(self, x):
        y_spatial = self.spatial_layer(x)
        # print('         spatial_layer', y_spatial.mean().item())
        y, rm, im = get_fft(x)
        # print('         *fft', y.mean().item(),
        #      rm[0].item(), rm[1].item(), im[0].item(), im[1].item())

        y = pad3d(y, (self.cropped, self.cropped, int(self.cropped/2) + 1))

        auto_filter = self.gate(y)
        # print('         auto_filter', auto_filter.mean().item())

        y = self.frequency_layer(y)
        # print('         freq_layer', y.mean().item())

        y = torch.cat((((y[:, :int(y.shape[1]/2)]*(rm[1]-rm[0]))+rm[0]) * auto_filter,
                      ((y[:, int(y.shape[1]/2):]*(im[1]-im[0]))+im[0]) * auto_filter), dim=1)
        # print('         freq_layer_filtered', y.mean().item())

        y = get_filtered_inverse_fft(y, shape=self.img_size)
        # print('         freq_back', y.mean().item())

        y = y_spatial + y
        y = self.norm(y)
        # print('         y_norm', y.mean().item())
        y = self.non_linearity(y)
        # print('         y_out', y.mean().item())
        y = self.drop(y)
        return y


class FourierBlock(nn.Module):
    def __init__(self, in_c, hid_c, out_c, img_size, dropout=0):
        super(FourierBlock, self).__init__()
        self.layers = nn.Sequential(FourierSpectralBlock(in_c, hid_c,
                                                         img_size, dropout),
                                    nn.Conv3d(hid_c, out_c, kernel_size=1),
                                    nn.InstanceNorm3d(out_c),
                                    nn.GELU(),
                                    nn.Dropout3d(dropout)
                                    )

    def forward(self, x):
        y = self.layers(x)
        return y
