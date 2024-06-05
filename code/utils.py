# -*- coding: utf-8 -*-
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


def pad3d(inpt, target):
    '''
    Pad or crop input image to match target size

    Parameters
    ----------
    inpt : torch.tensor
        Input tensor to be padded or cropped of shape (B, C, X, Y, Z).
    target : torch.tensor, tuple
        Target tensor of shape (B, C, X, Y, Z) or tuple of shape (X, Y, Z).

    Returns
    -------
    torch.tensor
        Resized (padded or cropped) input tensor matching size of target.

    '''
    if torch.is_tensor(target):
        delta = [target.shape[2+i] - inpt.shape[2+i] for i in range(3)]
    else:
        try:
            delta = [target[i] - inpt.shape[2+i] for i in range(3)]
        except Exception:
            delta = [target - inpt.shape[2+i] for i in range(3)]
    return nn.functional.pad(input=inpt, pad=(ceil(delta[2]/2),
                                              delta[2] - ceil(delta[2]/2),
                                              ceil(delta[1]/2),
                                              delta[1] - ceil(delta[1]/2),
                                              ceil(delta[0]/2),
                                              delta[0] - ceil(delta[0]/2)),
                             mode='constant', value=0.).to(dtype=inpt.dtype,
                                                           device=inpt.device)


def grad_penalty(critic, real, fake, weight):
    device = real.device
    b_size, c, h, w, d = real.shape
    epsilon = torch.rand(b_size, 1, 1, 1, 1).repeat(1, c, h, w, d).to(device)
    interp_img = (real * epsilon) + (fake * (1 - epsilon))

    mix_score = critic(interp_img)

    grad = torch.autograd.grad(outputs=mix_score,
                               inputs=interp_img,
                               grad_outputs=torch.ones_like(
                                   mix_score).to(device),
                               create_graph=True,
                               retain_graph=True)[0]

    grad = grad.view(b_size, -1)
    grad_norm = torch.sqrt((torch.sum(grad ** 2, dim=1)) + 1E-12)
    penalty = torch.mean((grad_norm - 1) ** 2)
    return weight * penalty


class GMELoss3D(nn.Module):
    '''
    Gradient Magnitude Edge Loss for 3D image data of shape (B, C, x, y, z)
    3D-Edge Loss for PyTorch with choice of criterion. Default is MSELoss.
    '''

    def __init__(self, n1=1, n2=2, n3=2, device='cpu'):
        super(GMELoss3D, self).__init__()
        self.edge_filter = GradEdge3D(n1, n2, n3, device)

    def forward(self, x, y):
        assert x.shape == y.shape, 'Inputs must be of the same shape'
        assert x.device == y.device, 'Inputs must be on the same device'
        edge_x = self.edge_filter.detect(x)
        edge_y = self.edge_filter.detect(y)
        return torch.mean((edge_x - edge_y) ** 2)


class Mask_L1Loss(nn.Module):
    def __init__(self, epsilon=1e-9):
        '''
        L1 Loss over masked region only
        '''
        super(Mask_L1Loss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y, mask):
        '''
        Compute the L1 loss over the masked region

        Parameters
        ----------
        x : torch tensor
            target ground truth.
        y : torch tensor
            prediction.
        mask : torch tensor
            binary mask.

        Returns
        -------
        torch tensor
            L1 error over masked region.
        '''
        assert x.shape == y.shape, 'Inputs must be of the same shape'
        assert x.shape == mask.shape, 'Mask must be of the same shape'
        assert x.device == y.device, 'Inputs must be on the same device'
        assert x.device == mask.device, 'Mask must be on the same device'
        mask = mask > 0.5
        error = torch.abs(x - y) * mask
        error = torch.sum(error) / (torch.sum(mask) + self.epsilon)
        return error


class Mask_MSELoss(nn.Module):
    def __init__(self, epsilon=1e-9):
        '''
        L1 Loss over masked region only
        '''
        super(Mask_MSELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, x, y, mask):
        '''
        Compute the L1 loss over the masked region

        Parameters
        ----------
        x : torch tensor
            target ground truth.
        y : torch tensor
            prediction.
        mask : torch tensor
            binary mask.

        Returns
        -------
        torch tensor
            L1 error over masked region.
        '''
        assert x.shape == y.shape, 'Inputs must be of the same shape'
        assert x.shape == mask.shape, 'Mask must be of the same shape'
        assert x.device == y.device, 'Inputs must be on the same device'
        assert x.device == mask.device, 'Mask must be on the same device'
        mask = mask > 0.5
        error = ((x - y) * mask) ** 2
        error = torch.sum(error) / (torch.sum(mask) + self.epsilon)
        return error


class PSNR_Metric(nn.Module):
    def __init__(self, max_psnr=100.0, epsilon=1E-9):
        """
        Initialize the PSNR metric class.

        Args:
        max_pixel_value (float): The maximum possible pixel value of the images.
                                 This depends on the scaling of the pixel values.
                                 For example, 1.0 for images scaled between 0 and 1,
                                 255 for typical 8-bit images, etc.
        """
        super(PSNR_Metric, self).__init__()
        self.max_psnr = torch.tensor(max_psnr)
        self.epsilon = epsilon

    def forward(self, x, y):
        """
        Compute the PSNR between two images.

        Args:
        x (torch.Tensor): The first image or batch of images.
        y (torch.Tensor): The second image or batch of images that serves as the ground truth.

        Returns:
        float: The PSNR value.
        """
        max_pixel_value = max([torch.max(x).item(), 1.])
        mse = torch.mean((x - y) ** 2)
        if mse == 0:
            return self.max_psnr
        psnr = 10 * (torch.log10(max_pixel_value**2 /
                     (torch.sqrt(mse) + self.epsilon)) + self.epsilon)
        return torch.min(psnr, self.max_psnr)


class SSIM_Metric(nn.Module):
    def __init__(self, channel=1, spatial_dims=3, win_size=11, win_sigma=1.5):
        super(SSIM_Metric, self).__init__()
        self.ssim = SSIM(channel=channel, spatial_dims=spatial_dims,
                         win_size=win_size, win_sigma=win_sigma)

    def forward(self, x, y):
        assert x.shape == y.shape, "inputs must be of same shape!"
        return self.ssim(x, y)


class ssim_loss(nn.Module):
    def __init__(self, channel=1, spatial_dims=3, win_size=11, win_sigma=1.5):
        super(ssim_loss, self).__init__()
        self.ssim = SSIM_Metric(channel=channel, spatial_dims=spatial_dims,
                                win_size=win_size, win_sigma=win_sigma)

    def forward(self, x, y):
        assert x.shape == y.shape, "inputs must be of same shape!"
        loss = 1 - self.ssim(x, y)
        return loss


def norm(x):
    EPSILON = 1E-9
    if torch.is_tensor(x):
        return (x - torch.min(x)) / ((torch.max(x) - torch.min(x)) + EPSILON)
    else:
        try:
            return (x - np.min(x)) / ((np.max(x) - np.min(x)) + EPSILON)
        except Exception:
            try:
                return [(i - min(x))/(max(x) - min(x)) for i in x]
            except Exception:
                print('WARNING: Input could not be normalized!')


def np_mse(y, yp):
    error = np.mean((y - yp)**2)
    return error


def plot_scans(scans=[], figsize=(15, 15), dpi=180, title=None):
    c = len(scans[0].shape)
    r = len(scans)
    i = 0

    fig = plt.figure(figsize=figsize, dpi=dpi)

    for scan in scans:
        scan = zoom(scan, [1, 1.2, 1], order=0)

        if i < 6:
            cmap = 'gray'
        elif i < 12:
            cmap = 'gray'  # 'jet'#'gray'
        else:
            cmap = 'viridis'  # 'gist_ncar'#'jet'

        a = scan[int(scan.shape[0]/2)]
        a[0, 0] = 0
        a[0, 1] = 1
        fig.add_subplot(r, c, i+1)
        plt.imshow(np.flip(a.T, axis=[0, 1]), cmap=cmap)
        plt.subplots_adjust(wspace=0.01, hspace=.05)
        plt.axis('off')

        a = scan[:, int(scan.shape[1]/2)]
        a[0, 0] = 0
        a[0, 1] = 1
        fig.add_subplot(r, c, i+2)
        plt.imshow(np.flip(a.T, axis=[0, 1]), cmap=cmap)
        plt.subplots_adjust(wspace=0.01, hspace=.05)
        plt.axis('off')

        a = scan[:, :, int(2 * scan.shape[2]/3)]
        a[0, 0] = 0
        a[0, 1] = 1
        fig.add_subplot(r, c, i+3)
        plt.imshow(np.flip(a.T, axis=[0, 1]), cmap=cmap)
        plt.subplots_adjust(wspace=0.01, hspace=.05)
        plt.axis('off')

        # plt.colorbar(fraction=0.1, pad=0.05)

        i += 3

    if title is not None:
        plt.suptitle(title)
    plt.show()


def show_images(data, num_samples=9, cols=3, masking=0, mask_signal=False,
                cmap='gray', fs=5, dpi=100):
    if mask_signal:
        data *= torch.where(data[masking] > 0, 1, 0)[None, ...]
    data = data[..., int(data.shape[-1]/2)]
    data = norm(data)
    plt.figure(figsize=(fs, fs), dpi=dpi)

    for i, img in enumerate(data):
        if i == num_samples:
            break
        plt.subplot(int(num_samples/cols) + 1, cols, i + 1)
        plt.axis('off')
        if img.shape[0] == 1:
            plt.imshow(img[0], cmap=cmap)
        else:
            plt.imshow(img.permute(1, 2, 0))
    plt.show()


def train_visualize(metrics, gans=False, dpi=200, HYAK=False):
    if gans:
        critic_losses_train, losses_train, losses_val, mae_val, mse_val, ssim_val, psnr_val = metrics

        plt.figure(dpi=dpi)
        plt.plot(norm(critic_losses_train), label='Critic Training Loss')
        plt.plot(norm(losses_train), label='Generator Training Loss')
        plt.plot(norm(losses_val), label='Generator Validation Loss')
        plt.title('Normalized Loss Curve')
        plt.xlabel('Epoch')
        plt.ylabel('Metrics')
        plt.legend()
        if HYAK:
            plt.savefig('/gscratch/kurtlab/brats-local-synthesis/plot0.png',
                        dpi=dpi, transparent=True, bbox_inches='tight')
        else:
            plt.show()

    else:
        losses_train, losses_val, mae_val, mse_val, ssim_val, psnr_val = metrics

    plt.figure(dpi=dpi)
    plt.plot(losses_train, label='Training Loss')
    plt.plot(losses_val, label='Validation Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Metrics')
    plt.legend()
    if HYAK:
        plt.savefig('/gscratch/kurtlab/brats-local-synthesis/plot1.png',
                    dpi=dpi, transparent=True, bbox_inches='tight')
    else:
        plt.show()

    plt.figure(dpi=dpi)
    plt.plot(norm(mae_val), label='-log(MAE)', color='grey')
    plt.plot(norm(mse_val), label='-log(MSE)', color='red')
    plt.plot(norm(ssim_val), label='-log(1-SSIM)',
             color='green', linestyle='--')
    plt.plot(norm(psnr_val), label='PSNR',
             color='blue', linestyle='--')
    plt.title('Normalized Validation Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Normalized Metric Score')
    plt.legend()
    if HYAK:
        plt.savefig('/gscratch/kurtlab/brats-local-synthesis/plot2.png',
                    dpi=dpi, transparent=True, bbox_inches='tight')
    else:
        plt.show()

    plt.figure(dpi=dpi)
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    axs[0, 0].plot(mae_val, label='-log(MAE)', color='grey')
    axs[0, 0].set_title('-log(MAE)')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Normalized Metric Score')
    axs[0, 0].legend()
    axs[0, 1].plot(mse_val, label='-log(MSE)', color='red')
    axs[0, 1].set_title('-log(MSE)')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Normalized Metric Score')
    axs[0, 1].legend()
    axs[1, 0].plot(ssim_val, label='-log(1-SSIM)',
                   color='green', linestyle='--')
    axs[1, 0].set_title('-log(1-SSIM)')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Normalized Metric Score')
    axs[1, 0].legend()
    axs[1, 1].plot(psnr_val, label='PSNR',
                   color='blue', linestyle='--')
    axs[1, 1].set_title('PSNR')
    axs[1, 1].set_xlabel('Epoch')
    axs[1, 1].set_ylabel('Normalized Metric Score')
    axs[1, 1].legend()
    plt.tight_layout()
    if HYAK:
        plt.savefig('/gscratch/kurtlab/brats-local-synthesis/plot3.png',
                    dpi=dpi, transparent=True, bbox_inches='tight')
    else:
        plt.show()
