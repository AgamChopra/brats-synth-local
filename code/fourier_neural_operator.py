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

from utils import count_parameters, test_model_memory_usage, pad3d


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


def test_vision_transformer3d():
    # Define the model parameters
    img_size = 48
    in_c = 1
    hid_c = 512
    out_c = 32

    # Instantiate the VisionTransformer3D model
    model = FourierBlock(in_c, hid_c, out_c, img_size)

    # Print the model architecture (optional)
    print(
        f'\nFNO Model size: {count_parameters(model)}\n'
    )
    # print(model)

    # Create a random input tensor with the shape (batch_size, channels, depth, height, width)
    batch_size = 1
    input_tensor = torch.randn(batch_size, in_c, img_size, img_size, img_size)

    # Pass the input tensor through the model
    output = model(input_tensor)

    # Print the output shape
    print("Output shape:", output.shape)

    test_model_memory_usage(model, input_tensor)

    print("Test passed!")


# Run the test function
if __name__ == '__main__':
    test_vision_transformer3d()
