#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 2024

@author: Agam Chopra
"""

import torch
import os
import nibabel as nib
from tqdm import trange

import dataloader_infer as dataloader
from models import Global_UNet


def save_nifti_with_origin(prediction, output_file, affine):
    # Create the NIfTI image with the given prediction data and affine matrix
    img = nib.Nifti1Image(prediction, affine)

    # Save the NIfTI image to the specified output file
    nib.save(img, output_file)


def crop_tensor(tensor, target_shape):
    """
    Perform a center crop on an N-dimensional tensor to a target shape.
    Args:
        tensor (torch.Tensor): The input tensor to crop.
        target_shape (tuple): The target shape to crop to.
    Returns:
        torch.Tensor: The center-cropped tensor.
    """
    # Calculate the starting indices for the center crop
    start_indices = [(dim - target_dim) // 2 for dim,
                     target_dim in zip(tensor.shape, target_shape)]

    # Calculate the slices for each dimension
    slices = tuple(slice(start, start + target)
                   for start, target in zip(start_indices, target_shape))

    # Crop the tensor
    cropped_tensor = tensor[slices]

    return cropped_tensor


def match_contrast_inpaint(source, reference, mask):
    """
    Match the contrast of the predicted region to that of the surrounding region.
    Args:
        predicted_region (torch.Tensor): Predicted region tensor of shape (D, H, W)
        surrounding_region (torch.Tensor): Surrounding region tensor of shape (D, H, W)
    Returns:
        torch.Tensor: Adjusted predicted region tensor.
    """
    whole_mask = torch.where(reference > 0, 1, 0) + mask
    surrounding_region_mask = whole_mask * (crop_tensor(
        torch.nn.functional.interpolate(mask, scale_factor=1.1),
        target_shape=mask.shape) - mask)
    predicted_region = source[mask.bool()]
    surrounding_region = reference[(surrounding_region_mask).bool()]

    # Calculate mean and std of both regions
    mean_pred = predicted_region.mean()
    std_pred = predicted_region.std()
    mean_surround = surrounding_region.mean()
    std_surround = surrounding_region.std()

    # Adjust predicted region to match surrounding contrast
    adjusted_region = (predicted_region - mean_pred) / \
        std_pred * std_surround + mean_surround
    source[mask.bool()] = adjusted_region
    return source * mask, surrounding_region_mask


def run_inference(data_path, output_path, weights):
    # load model
    model = Global_UNet(
        in_c=1,
        out_c=1,
        fact=32,
        embed_dim=256,
        n_heads=16,
        mlp_ratio=32,
        qkv_bias=True,
        dropout_rate=0.,
        mask_downsample=16,
        noise=True,
        device1='cpu',
        device2='cpu'
    )
    state_dict = torch.load(weights, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()

    # load prepared data
    # dataloader should output cropped image, mask, and image_id_folder_name
    # (batch size 1, no augmentation)
    data = dataloader.DataLoader(workers=4, norm=True, path=data_path)
    iterations = data.max_id

    # inference
    with torch.no_grad():
        for _ in trange(iterations):
            # Load batch and information
            input_image, mask, file_id, affine = data.load_batch()
            whole_mask = (input_image > 0.001).float() + mask

            # Predictions
            output = model(input_image+whole_mask*mask *
                           torch.clip(torch.rand_like(input_image), 0, 1),
                           mask) * whole_mask * mask
            output = match_contrast_inpaint(
                output, input_image, mask * whole_mask)[0]
            output = torch.clip(output * whole_mask * mask,
                                input_image.min(), input_image.max())
            output += input_image

            # Remove the first two singleton dimensions to
            # get the shape (x, y, z)
            # Assuming output is of shape (1, 1, x, y, z)
            print(output.shape)
            output = output.detach().squeeze().numpy()

            # Output path
            out_path = os.path.join(
                output_path, f"{file_id}-t1n-inference.nii.gz")
            print(out_path)

            # Save the image to a .nii.gz file
            save_nifti_with_origin(output, out_path, affine)
