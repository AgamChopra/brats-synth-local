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

from math import ceil
import numpy as np
import torch
import torch.nn as nn
from pytorch_msssim import SSIM
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from edge_loss import GradEdge3D
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import time


def get_memory_usage():
    # Initialize NVML
    nvmlInit()

    # Get handle for the first GPU
    handle = nvmlDeviceGetHandleByIndex(0)

    # Get memory info
    info = nvmlDeviceGetMemoryInfo(handle)

    return (info.total / 1024**2, info.free / 1024**2, info.used / 1024**2)


def test_model_memory_usage(model, input_tensor):
    # Move model and input to GPU
    (t, f, u) = get_memory_usage()
    print(f"Total memory: {t} MB")
    print(f"Free memory: {f} MB")
    print(f"Used memory: {u} MB")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Measure memory usage before forward pass
    (t, f, u1) = get_memory_usage()
    print(f"Memory used by model before forward pass: {u1 - u} MB")

    # Perform forward pass
    start_time = time.time()
    output = model(input_tensor)

    # Measure memory usage after forward pass
    (t, f, u2) = get_memory_usage()
    print(f"Memory used by model after forward pass: {u2 - u} MB")

    # Perform backward pass
    output.sum().backward()

    # Measure memory usage after backward pass
    (t, f, u3) = get_memory_usage()
    print(f"Memory used by model after backward pass: {u3 - u} MB")

    # Perform optimization step
    optimizer.step()
    optimizer.zero_grad()
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Measure memory usage after optimization step
    (t, f, u4) = get_memory_usage()
    print(f"Memory used by model after optimization step: {u4 - u} MB")

    # Print output shape and elapsed time
    print("Elapsed time: {:.6f} seconds".format(elapsed_time))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def get_fft(x, cropped=32):
    fft_image = torch.fft.rfftn(x)
    fft_shifted = torch.fft.fftshift(fft_image)

    real = fft_shifted.real
    imag = fft_shifted.imag

    real = pad3d(real, (cropped, cropped, int(cropped/2) + 1))
    imag = pad3d(imag, (cropped, cropped, int(cropped/2) + 1))

    return torch.cat((real, imag), dim=1)


def reconst_image(real_imag, shape=240):
    real_imag = pad3d(real_imag, (shape, shape, int(shape/2) + 1))

    fft_reconstructed = torch.complex(real_imag[:, 0:1], real_imag[:, 1:2])

    ifft_shifted = torch.fft.ifftshift(fft_reconstructed)
    reconstructed_image = torch.fft.irfftn(
        ifft_shifted, s=(shape, shape, shape)).real

    return reconstructed_image


# =============================================================================
# def get_fft(x, cropped=32):
#     fft_image = torch.fft.rfftn(x)
#     fft_shifted = torch.fft.fftshift(fft_image)
#     #real = fft_shifted.real
#     #imag = fft_shifted.imag
#
#     # (torch.log(1 + torch.sqrt(real**2 + imag**2)) / 6.5) - 1
#     mag = (torch.log(1 + torch.abs(fft_shifted)) / 6.5) - 1
#     phase = torch.angle(fft_shifted) / 3  # torch.atan2(imag, real) / 3
#
#     mag = pad3d(mag, (cropped, cropped, int(cropped/2) + 1))
#     phase = pad3d(phase, (cropped, cropped, int(cropped/2) + 1))
#
#     return torch.cat((mag, phase), dim=1)
#
#
# def reconst_image(mag_phase, shape=240):
#     mag_phase = pad3d(mag_phase, (shape, shape, int(shape/2) + 1))
#     mag, phase = mag_phase[:, 0:1], mag_phase[:, 1:2]
#
#     mag_ = (torch.exp((mag + 1) * 6.5) - 1)
#     phase_ = 3 * phase
#
#     real = mag_ * torch.cos(phase_)
#     imag = mag_ * torch.sin(phase_)
#
#     fft_reconstructed = torch.complex(real, imag)
#
#     ifft_shifted = torch.fft.ifftshift(fft_reconstructed)
#     reconstructed_image = torch.fft.irfftn(
#         ifft_shifted, s=(shape, shape, shape)).real
#
#     return reconstructed_image
# =============================================================================


def grad_penalty(critic, real, fake, weight):
    """
    Compute gradient penalty for WGAN-GP.

    Args:
        critic (nn.Module): The critic model.
        real (torch.tensor): Real images.
        fake (torch.tensor): Fake images generated by the generator.
        weight (float): Weight for the gradient penalty.

    Returns:
        torch.tensor: The gradient penalty.
    """
    device = real.device
    b_size, c, h, w, d = real.shape
    epsilon = torch.rand(b_size, 1, 1, 1, 1, device=device)
    epsilon = epsilon.expand_as(real)
    epsilon = epsilon.requires_grad_(True)

    interp_img = epsilon * real + (1 - epsilon) * fake
    interp_img = interp_img.requires_grad_(True)

    mix_score = critic(interp_img)
    grad = torch.autograd.grad(
        outputs=mix_score,
        inputs=interp_img,
        grad_outputs=torch.ones_like(mix_score).to(device),
        create_graph=True, retain_graph=True
    )[0]

    grad = grad.view(b_size, -1)
    grad_norm = torch.sqrt(torch.sum(grad ** 2, dim=1) + 1E-12)
    penalty = torch.mean((grad_norm - 1) ** 2)
    return weight * penalty


class GMELoss3D(nn.Module):
    """
    Gradient Magnitude Edge Loss for 3D image data of shape (B, C, x, y, z).

    Args:
        n1 (int): Filter size for the first dimension.
        n2 (int): Filter size for the second dimension.
        n3 (int): Filter size for the third dimension.
        device (str): Device to run the loss computation on.
    """

    def __init__(self, n1=1, n2=2, n3=2):
        super(GMELoss3D, self).__init__()
        self.edge_filter = GradEdge3D(n1, n2, n3)

    def forward(self, x, y):
        assert x.shape == y.shape, 'Inputs must be of the same shape'
        assert x.device == y.device, 'Inputs must be on the same device'
        edge_x = self.edge_filter.detect(x)
        edge_y = self.edge_filter.detect(y)
        return torch.mean((edge_x - edge_y) ** 2)


class Mask_L1Loss(nn.Module):
    """
    L1 Loss over masked region only.

    Args:
        epsilon (float, optional): Small value to avoid division by zero. Default is 1e-9.
    """

    def __init__(self, epsilon=1e-9):
        super(Mask_L1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y, mask):
        """
        Compute the L1 loss over the masked region.

        Args:
            x (torch.tensor): Target ground truth.
            y (torch.tensor): Prediction.
            mask (torch.tensor): Binary mask.

        Returns:
            torch.tensor: L1 error over masked region.
        """
        assert x.shape == y.shape, 'Inputs must be of the same shape'
        assert x.shape == mask.shape, 'Mask must be of the same shape'
        assert x.device == y.device, 'Inputs must be on the same device'
        assert x.device == mask.device, 'Mask must be on the same device'
        mask = mask > 0.5
        error = torch.abs(x - y) * mask
        error = torch.sum(error) / (torch.sum(mask) + self.epsilon)
        return error


class Mask_MSELoss(nn.Module):
    """
    MSE Loss over masked region only.

    Args:
        epsilon (float, optional): Small value to avoid division by zero. Default is 1e-9.
    """

    def __init__(self, epsilon=1e-9):
        super(Mask_MSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y, mask):
        """
        Compute the MSE loss over the masked region.

        Args:
            x (torch.tensor): Target ground truth.
            y (torch.tensor): Prediction.
            mask (torch.tensor): Binary mask.

        Returns:
            torch.tensor: MSE error over masked region.
        """
        assert x.shape == y.shape, 'Inputs must be of the same shape'
        assert x.shape == mask.shape, 'Mask must be of the same shape'
        assert x.device == y.device, 'Inputs must be on the same device'
        assert x.device == mask.device, 'Mask must be on the same device'
        mask = mask > 0.5
        error = ((x - y) * mask) ** 2
        error = torch.sum(error) / (torch.sum(mask) + self.epsilon)
        return error


class PSNR_Metric(nn.Module):
    """
    PSNR (Peak Signal-to-Noise Ratio) metric for image quality assessment.

    Args:
        max_psnr (float, optional): Maximum PSNR value. Default is 100.0.
        epsilon (float, optional): Small value to avoid division by zero. Default is 1E-9.
    """

    def __init__(self, max_psnr=100.0, epsilon=1E-9):
        super(PSNR_Metric, self).__init__()
        self.max_psnr = torch.tensor(max_psnr)
        self.epsilon = epsilon

    def forward(self, x, y):
        """
        Compute the PSNR between two images.

        Args:
            x (torch.tensor): The first image or batch of images.
            y (torch.tensor): The second image or batch of images that serves as the ground truth.

        Returns:
            float: The PSNR value.
        """
        max_pixel_value = max([torch.max(x).item(), 1.0])
        mse = torch.mean((x - y) ** 2)
        if mse == 0:
            return self.max_psnr
        psnr = 10 * (torch.log10(max_pixel_value ** 2 /
                     (torch.sqrt(mse) + self.epsilon)) + self.epsilon)
        return torch.min(psnr, self.max_psnr)


class SSIM_Metric(nn.Module):
    """
    SSIM (Structural Similarity Index Measure) metric for image quality assessment.

    Args:
        channel (int, optional): Number of channels. Default is 1.
        spatial_dims (int, optional): Number of spatial dimensions. Default is 3.
        win_size (int, optional): Size of the window. Default is 11.
        win_sigma (float, optional): Standard deviation of the Gaussian window. Default is 1.5.
    """

    def __init__(self, channel=1, spatial_dims=3, win_size=11, win_sigma=1.5):
        super(SSIM_Metric, self).__init__()
        self.ssim = SSIM(channel=channel, spatial_dims=spatial_dims,
                         win_size=win_size, win_sigma=win_sigma)

    def forward(self, x, y):
        assert x.shape == y.shape, "Inputs must be of the same shape!"
        return self.ssim(x, y)


class SSIMLoss(nn.Module):
    """
    SSIM Loss function for image quality assessment.

    Args:
        channel (int, optional): Number of channels. Default is 1.
        spatial_dims (int, optional): Number of spatial dimensions. Default is 3.
        win_size (int, optional): Size of the window. Default is 11.
        win_sigma (float, optional): Standard deviation of the Gaussian window. Default is 1.5.
    """

    def __init__(self, channel=1, spatial_dims=3, win_size=11, win_sigma=1.5):
        super(SSIMLoss, self).__init__()
        self.ssim = SSIM_Metric(
            channel=channel, spatial_dims=spatial_dims, win_size=win_size, win_sigma=win_sigma)

    def forward(self, x, y):
        assert x.shape == y.shape, "Inputs must be of the same shape!"
        loss = 1 - self.ssim(x, y)
        return loss


def norm(x):
    """
    Normalize the input tensor or array.

    Args:
        x (torch.tensor or np.array): Input data to be normalized.

    Returns:
        torch.tensor or np.array: Normalized data.
    """
    EPSILON = 1E-9
    if torch.is_tensor(x):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x) + EPSILON)
    else:
        try:
            return (x - np.min(x)) / (np.max(x) - np.min(x) + EPSILON)
        except Exception:
            try:
                return [(i - min(x)) / (max(x) - min(x)) for i in x]
            except Exception:
                print('WARNING: Input could not be normalized!')


def np_mse(y, yp):
    """
    Compute the Mean Squared Error between two numpy arrays.

    Args:
        y (np.array): Ground truth array.
        yp (np.array): Predicted array.

    Returns:
        float: Mean Squared Error.
    """
    return np.mean((y - yp) ** 2)


def plot_scans(scans=[], figsize=(15, 15), dpi=180, title=None):
    """
    Plot 3D scans with given configurations.

    Args:
        scans (list): List of 3D scans to plot.
        figsize (tuple, optional): Size of the figure. Default is (15, 15).
        dpi (int, optional): Dots per inch for the figure. Default is 180.
        title (str, optional): Title of the figure. Default is None.
    """
    c = len(scans[0].shape)
    r = len(scans)
    i = 0

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for scan in scans:
        scan = zoom(scan, [1, 1.2, 1], order=0)

        cmap = 'gray' if i < 12 else 'viridis'
        for j in range(3):
            a = scan.take(indices=int((j + 1) * scan.shape[j] / 3), axis=j)
            a[0, 0], a[0, 1] = 0, 1
            fig.add_subplot(r, c, i + j + 1)
            plt.imshow(np.flip(a.T, axis=[0, 1]), cmap=cmap)
            plt.axis('off')
            plt.subplots_adjust(wspace=0.01, hspace=0.05)
        i += 3

    if title:
        plt.suptitle(title)
    plt.show()


def show_images(data, num_samples=9, cols=3, masking=0, mask_signal=False, cmap='gray', fs=(5, 5), dpi=100):
    """
    Display a set of images.

    Args:
        data (torch.tensor): Data to be displayed.
        num_samples (int, optional): Number of samples to display. Default is 9.
        cols (int, optional): Number of columns in the display grid. Default is 3.
        masking (int, optional): Masking index. Default is 0.
        mask_signal (bool, optional): Flag to apply masking. Default is False.
        cmap (str, optional): Colormap for the images. Default is 'gray'.
        fs (int, optional): Figure size. Default is 5.
        dpi (int, optional): Dots per inch for the figure. Default is 100.
    """
    if mask_signal:
        data *= torch.where(data[masking] > 0, 1, 0)[None, ...]
    data = data[..., int(data.shape[-1] / 2)]
    data = norm(data)
    plt.figure(figsize=(fs[0], fs[1]), dpi=dpi)

    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples / cols) + 1, cols, i + 1)
        plt.axis('off')
        plt.imshow(img[0] if img.shape[0] ==
                   1 else img.permute(1, 2, 0), cmap=cmap)
    plt.show()


def train_visualize(metrics, gans=False, dpi=200, path=None, identity=''):
    """
    Visualize training metrics.

    Args:
        metrics (list): List of training metrics.
        gans (bool, optional): Flag indicating GAN training. Default is False.
        dpi (int, optional): Dots per inch for the figure. Default is 200.
        HYAK (bool, optional): Flag for saving figures on HYAK. Default is False.
    """
    if gans:
        critic_losses_train, losses_train, losses_val, mae_val, mse_val, ssim_val, psnr_val = metrics

        plt.figure(dpi=dpi)
        plt.plot(losses_train, label='Generator Training Error')
        plt.plot(losses_val[0], label='Critic Training Real Error')
        plt.plot(losses_val[1], label='Critic Training Fake Error')
        plt.plot(critic_losses_train, label='Critic Training Error')
        plt.title('GANs Error')
        plt.xlabel('Epoch')
        plt.ylabel('Norm Error')
        plt.legend()
        save_plot(filepath=f'{path}{identity}_gans_loss.png' if path else None)

    else:
        losses_train, losses_val, mae_val, mse_val, ssim_val, psnr_val = metrics

    plt.figure(dpi=dpi)
    plt.plot(losses_train, label='Training Error')
    plt.plot(losses_val, label='Validation Error')
    plt.title('Synthesis Error')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    save_plot(filepath=f'{path}{identity}_synth_loss.png' if path else None)

    plt.figure(dpi=dpi)
    plt.plot(norm(mae_val), label='-log(MAE)', color='grey')
    plt.plot(norm(mse_val), label='-log(MSE)', color='red')
    plt.plot(norm(ssim_val), label='-log(1-SSIM)',
             color='green', linestyle='--')
    plt.plot(norm(psnr_val), label='PSNR', color='blue', linestyle='--')
    plt.title('Normalized Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Score')
    plt.legend()
    save_plot(
        filepath=f'{path}{identity}_norm_validation_metrics.png' if path else None)

    plt.figure(dpi=dpi)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10), dpi=dpi)
    plot_metric(axs[0, 0], mae_val, '-log(MAE)', 'grey')
    plot_metric(axs[0, 1], mse_val, '-log(MSE)', 'red')
    plot_metric(axs[1, 0], ssim_val, '-log(1-SSIM)', 'green', '--')
    plot_metric(axs[1, 1], psnr_val, 'PSNR', 'blue', '--')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.tight_layout()
    save_plot(
        filepath=f'{path}{identity}_validation_metrics.png' if path else None)


def save_plot(filepath=None, transparent=False, dpi=200):
    """
    Save the plot to a file if HYAK flag is set, otherwise show the plot.

    Args:
        filepath (str): Path to save the plot.
        HYAK (bool): Flag indicating if the plot should be saved.
    """
    if filepath is not None:
        plt.savefig(filepath, dpi=dpi, transparent=transparent,
                    bbox_inches='tight')
    else:
        plt.show()


def plot_metric(ax, metric, title, color, linestyle='-'):
    """
    Plot a single metric on a given axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on.
        metric (list): Metric values to plot.
        title (str): Title of the plot.
        color (str): Color of the plot line.
        linestyle (str, optional): Line style of the plot. Default is '-'.
    """
    ax.plot(metric, label=title, color=color, linestyle=linestyle)
    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Normalized Metric Score')
    ax.legend()
