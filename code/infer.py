#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 2024

@author: Agam Chopra
"""

import torch
import os
import nibabel as nib
import numpy as np
from tqdm import trange

import dataloader_infer as dataloader
from models import Global_UNet
from utils import match_contrast_inpaint


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
    data = dataloader.DataLoader_infer(workers=4, norm=True, path=data_path)
    iterations = data.max_id

    # inference
    with torch.no_grad():
        for _ in trange(iterations):
            # Load batch and information
            input_image, mask, file_id = data.load_batch()
            
            # Predictions
            output = model(input_image, mask)
            output = match_contrast_inpaint(output, input_image, mask)
            
            # Remove the first two singleton dimensions to
            # get the shape (x, y, z)
            # Assuming output is of shape (1, 1, x, y, z)
            output = output.detach().squeeze().numpy()
            
            # Output path
            output_path = os.path.join(output_path, f"{file_id}.nii.gz")
            
            # Create a NIfTI image
            nifti_output = nib.Nifti1Image(output, affine=np.eye(4))
            
            # Save the image to a .nii.gz file
            nib.save(nifti_output, output_path)
            
            
    
