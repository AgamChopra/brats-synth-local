import torch
from dataloader import DataLoader, show_images
from utils import pad3d
from tqdm import trange


def get_fft_mag_phase(x, cropped=32):
    fft_image = torch.fft.rfftn(x)
    fft_shifted = torch.fft.fftshift(fft_image)
    print(fft_shifted.shape)
    real = fft_shifted.real
    imag = fft_shifted.imag

    mag = (torch.log(1 + torch.sqrt(real**2 + imag**2)) / 6.5) - 1
    phase = torch.atan2(imag, real) / 3

    mag = pad3d(mag, (cropped, cropped, cropped + 1))
    phase = pad3d(phase, (cropped, cropped, cropped + 1))

    return torch.cat((mag, phase), dim=1)


def reconst_image(mag_phase, shape=240):
    mag_phase = pad3d(mag_phase, (shape, shape, int(shape/2) + 1))
    mag, phase = mag_phase[:, 0:1], mag_phase[:, 1:2]

    mag_ = (torch.exp((mag + 1) * 6.5) - 1)
    phase_ = 3 * phase

    real = mag_ * torch.cos(phase_)
    imag = mag_ * torch.sin(phase_)

    fft_reconstructed = torch.complex(real, imag)

    ifft_shifted = torch.fft.ifftshift(fft_reconstructed)
    reconstructed_image = torch.fft.irfftn(
        ifft_shifted, s=(shape, shape, shape)).real

    return reconstructed_image


def apply_fft(x):
    mag_phase = get_fft_mag_phase(x, 32)

    reconstructed_image = reconst_image(mag_phase, x.shape[-1])

    return reconstructed_image


def test_fft(N=2000):
    error = 0.
    loader = DataLoader(augment=True, batch=1, workers=4,
                        path='/home/agam/Desktop/brats_2024_local_impainting/TrainingData/')
    for i in trange(N):
        x, _ = loader.load_batch()
        x = pad3d(x[0:2], 240)
        x_ = apply_fft(x)
        L1_distance = torch.abs(x - x_)

        show_images(torch.cat((torch.permute(x, (0, 1, 4, 2, 3)),
                               torch.permute(x, (0, 1, 4, 3, 2)),
                               torch.permute(x_, (0, 1, 4, 2, 3)),
                               torch.permute(x_, (0, 1, 4, 3, 2)),
                               torch.permute(L1_distance, (0, 1, 4, 2, 3)),
                               torch.permute(L1_distance, (0, 1, 4, 3, 2))),
                              dim=0), 6, 2, dpi=250, fs=(2, 4))
        error += torch.mean(L1_distance).item()
    error /= N
    print(f'average error:{error}')


if __name__ == '__main__':
    test_fft(100)
