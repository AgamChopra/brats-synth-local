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


def get_fft(x):
    max_value = torch.finfo(torch.float16).max / 1.1
    min_value = -max_value

    fft_image = torch.fft.rfftn(x)
    fft_shifted = torch.fft.fftshift(fft_image)
    real = fft_shifted.real
    imag = fft_shifted.imag

    real = torch.clamp(real, min=min_value, max=max_value)
    imag = torch.clamp(imag, min=min_value, max=max_value)

    real = torch.nan_to_num(real, nan=0.0, posinf=max_value, neginf=min_value)
    imag = torch.nan_to_num(imag, nan=0.0, posinf=max_value, neginf=min_value)

    real_ = [real.min(), real.max()]
    imag_ = [imag.min(), imag.max()]
    real = (real - real_[0]) / (real_[1] - real_[0])
    imag = (imag - imag_[0]) / (imag_[1] - imag_[0])
    return torch.cat((real, imag), dim=1), real_, imag_


def get_filtered_inverse_fft(freq, crop_ratio=0.1):
    max_value = torch.finfo(torch.float16).max / 1.1
    min_value = -max_value

    shape = freq.shape[2]
    cropped = int(shape * crop_ratio)
    real_imag = pad3d(pad3d(freq, (cropped, cropped, int(
        cropped/2) + 1)), (shape, shape, int(shape/2) + 1))
    real, imag = real_imag[:, :int(
        freq.shape[1]/2)], real_imag[:, int(freq.shape[1]/2):]

    real = torch.clamp(real, min=min_value, max=max_value)
    imag = torch.clamp(imag, min=min_value, max=max_value)

    real = torch.nan_to_num(real, nan=0.0, posinf=max_value, neginf=min_value)
    imag = torch.nan_to_num(imag, nan=0.0, posinf=max_value, neginf=min_value)

    fft_reconstructed = torch.complex(real, imag)
    ifft_shifted = torch.fft.ifftshift(fft_reconstructed)
    reconstructed_image = torch.fft.irfftn(
        ifft_shifted, s=(shape, shape, shape)).real

    reconstructed_image = torch.clamp(
        reconstructed_image, min=min_value, max=max_value)
    reconstructed_image = torch.nan_to_num(
        reconstructed_image, nan=0.0, posinf=max_value, neginf=min_value)
    return reconstructed_image


class FourierNeuralOperator(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.):
        super(FourierNeuralOperator, self).__init__()
        self.spatial_layer = nn.Conv3d(
            in_features, out_features, kernel_size=1, stride=1,
            groups=in_features)

        self.frequency_layer = nn.Conv3d(
            in_features*2, out_features*2, kernel_size=1, stride=1,
            groups=in_features*2)

        self.non_linearity = nn.GELU()
        self.norm = nn.InstanceNorm3d(out_features*2)
        self.drop = nn.Dropout(dropout)

        self.gate = nn.Sequential(
            nn.Conv3d(
                in_features*2, out_features, kernel_size=1, stride=1),
            nn.InstanceNorm3d(out_features),
            nn.Sigmoid()
        )

    def forward(self, x, crop_ratio=1.):
        y_spatial = self.spatial_layer(x)
        print('         spatial_layer', y_spatial.mean().item())
        y, rm, im = get_fft(x)
        print('         *fft', y.mean().item(),
              rm[0].item(), rm[1].item(), im[0].item(), im[1].item())

        auto_filter = self.gate(y)
        print('         auto_filter', auto_filter.mean().item())

        y = self.frequency_layer(y)
        print('         freq_layer', y.mean().item())

        y = torch.cat((((y[:, :int(y.shape[1]/2)]*(rm[1]-rm[0]))+rm[0]) * auto_filter,
                      ((y[:, int(y.shape[1]/2):]*(im[1]-im[0]))+im[0]) * auto_filter), dim=1)
        print('         freq_layer_filtered', y.mean().item())

        y = get_filtered_inverse_fft(y, crop_ratio=crop_ratio)
        print('         freq_back', y.mean().item())

        y = torch.cat((y_spatial, y), dim=1)
        y = self.norm(y)
        print('         y_norm', y.mean().item())
        y = self.non_linearity(y)
        y = self.drop(y)
        print('         y_out', y.mean().item())
        return y


class FNOBlock(nn.Module):
    def __init__(self, in_c, hid_c, out_c, dropout=0):
        super(FNOBlock, self).__init__()
        self.layers = nn.Sequential(FourierNeuralOperator(in_c, hid_c, dropout),
                                    nn.Conv3d(hid_c*2, out_c, kernel_size=1)
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
    model = FNOBlock(in_c, hid_c, out_c)

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
