"""
Created on June 2023, Modified on May 2024
@author: Agamdeep Chopra
@email: achopra4@uw.edu
@website: https://agamchopra.github.io/
@affiliation: KurtLab, Department of Mechanical Engineering,
              University of Washington, Seattle, USA
@Refs:
"""
import os
import concurrent.futures
import torch
import nibabel as nib
import numpy as np


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


def load_patient(path, filename, nrm=True):
    """
    Load patient data from NIfTI file and optionally normalize it.

    Args:
        path (str): Path to the directory containing the patient data.
        filename (str): Name of the patient file.
        nrm (bool, optional): Whether to normalize the data. Default is True.

    Returns:
        torch.tensor: Loaded patient data tensor.
    """
    nib_image = nib.load(os.path.join(path, os.path.join(
        filename, filename + '-t1n-voided.nii.gz')))
    affine = nib_image.affine
    image = torch.from_numpy(nib_image.get_fdata())[None, None, ...].float()
    mask = torch.from_numpy(nib.load(os.path.join(path, os.path.join(
        filename, filename + '-mask.nii.gz'))).get_fdata())[None, None, ...].float()
    if nrm:
        image = norm(image)
    return (image, mask, filename, affine)


def list_folders(path):
    """
    List all folders in the specified directory.

    Args:
        path (str): Path to the directory.

    Returns:
        list: List of folder names.
    """
    folder_list = []
    try:
        with os.scandir(path) as entries:
            for entry in entries:
                if entry.is_dir():
                    folder_list.append(entry.name)
            return folder_list
    except FileNotFoundError:
        print("The specified path does not exist.")
        return folder_list
    except PermissionError:
        print("You do not have permissions to access the directory.")
        return folder_list


class DataLoader:
    """
    DataLoader for loading and augmenting batches of patient data.

    Args:
        batch (int, optional): Batch size. Default is 1.
        augment (bool, optional): Whether to apply augmentation. Default is True.
        aug_thresh (float, optional): Threshold for applying augmentation. Default is 0.05.
        workers (int, optional): Number of worker threads. Default is 4.
        norm (bool, optional): Whether to normalize the data. Default is True.
        path (str, optional): Path to the directory containing the data. Default is '/home/agam/Desktop/brats_2024_local_impainting/TrainingData/'.
    """

    def __init__(self, workers=4, norm=True,
                 path='/home/agam/Desktop/brats_2024_local_impainting/TrainingData/'):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers)
        self.id = 0
        self.batch = 1
        self.path = path
        self.samples = list_folders(self.path)
        self.nrm = norm
        sample_len = len(self.samples)
        self.max_id = sample_len  # !!!
        self.future_batch = None
        self.pre_fetch_next_batch()

    def pre_fetch_next_batch(self):
        """
        Pre-fetch the next batch of samples asynchronously.
        """
        if self.id + self.batch > self.max_id:
            new_samples = self.samples[self.id:] if self.id < self.max_id else self.samples[self.id:self.id + 1]
            self.id = 0
        else:
            new_samples = self.samples[self.id:self.id + self.batch]
            self.id += self.batch

        self.future_batch = self.executor.submit(
            self.load_batch_dataset_async, new_samples)

    def load_batch_dataset_async(self, sample_list):
        """
        Load a batch of samples asynchronously.

        Args:
            sample_list (list): List of sample filenames.

        Returns:
            torch.tensor: Batch of loaded samples.
        """
        futures = [self.executor.submit(
            load_patient, self.path, sample, self.nrm) for sample in sample_list]
        image_mask_filename_affine = [future.result() for future in futures]
        return image_mask_filename_affine

    def load_batch(self):
        """
        Load a batch of samples and apply augmentation and masking.

        Returns:
            tuple: Batch of raw samples and corresponding masks.
        """
        sample = self.future_batch.result()[0]
        self.pre_fetch_next_batch()
        image, mask, filename, affine = sample[0], sample[1], sample[2], sample[3]

        print(image.shape)
        print(mask.shape)
        print(filename)
        print(affine)

        return image, mask, filename, affine
