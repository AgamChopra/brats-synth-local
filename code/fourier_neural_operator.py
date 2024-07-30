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


def get_fft(x, device='cuda'):
    with torch.amp.autocast(device_type=device, enabled=False):
        x = x.float()  # Ensure input is float32 to avoid ComplexHalf issues
        fft_image = torch.fft.rfftn(x, dim=(-3, -2, -1))
        fft_shifted = torch.fft.fftshift(fft_image, dim=(-3, -2, -1))
        real = fft_shifted.real
        imag = fft_shifted.imag
    return torch.cat((real, imag), dim=1)


def get_filtered_inverse_fft(freq, crop_ratio=0.1, device='cuda'):
    with torch.amp.autocast(device_type=device, enabled=False):
        shape = freq.shape[2]
        cropped = int(shape * crop_ratio)

        real_imag = pad3d(pad3d(freq, (cropped, cropped, int(
            cropped/2) + 1)), (shape, shape, int(shape/2) + 1))

        real, imag = real_imag[:, :int(
            freq.shape[1]/2)], real_imag[:, int(freq.shape[1]/2):]

        real = real.float()
        imag = imag.float()

        fft_reconstructed = torch.complex(real, imag)

        ifft_shifted = torch.fft.ifftshift(fft_reconstructed, dim=(-3, -2, -1))
        reconstructed_image = torch.fft.irfftn(
            ifft_shifted, s=(shape, shape, shape)).real

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
        y = get_fft(x)

        auto_filter = self.gate(y)

        y = self.frequency_layer(y)

        y = torch.cat((y[:, :int(y.shape[1]/2)] * auto_filter,
                      y[:, int(y.shape[1]/2):] * auto_filter), dim=1)

        y = get_filtered_inverse_fft(y, crop_ratio=crop_ratio)

        y = torch.cat((y_spatial, y), dim=1)
        y = self.norm(y)
        y = self.non_linearity(y)
        y = self.drop(y)
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
