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
import torchio as tio
import torch.nn.functional as F
import random
import nibabel as nib

from utils import norm, show_images


def slice_augment(x, min_slices=12):
    """
    Perform slice augmentation by downsampling and then upsampling the input tensor.

    Args:
        x (torch.tensor): Input tensor of shape (B, C, D, H, W).
        min_slices (int, optional): Minimum number of slices for augmentation. Default is 12.

    Returns:
        torch.tensor: Augmented tensor.
    """
    B, C, D, H, W = x.shape
    target_depth = torch.randint(min_slices, D + 1, (1,)).item()
    target_height = torch.randint(min_slices, H + 1, (1,)).item()
    target_width = torch.randint(min_slices, W + 1, (1,)).item()

    downsampled = F.interpolate(x, size=(target_depth, target_height, target_width), mode='trilinear', align_corners=False)
    augmented = F.interpolate(downsampled, size=(D, H, W), mode='nearest')

    return augmented


def get_box_mask(x):
    """
    Generate a random box mask for the input tensor.

    Args:
        x (torch.tensor): Input tensor of shape (B, C, D, H, W).

    Returns:
        torch.tensor: Box mask tensor of the same shape as input.
    """
    B, C, D, H, W = x.shape
    d = torch.randint(D // 4, D // 2, (1,)).item()
    h = torch.randint(H // 4, H // 2, (1,)).item()
    w = torch.randint(W // 4, W // 2, (1,)).item()
    mask = torch.zeros_like(x)
    d_start = torch.randint(0, D - d, (1,)).item()
    h_start = torch.randint(0, H - h, (1,)).item()
    w_start = torch.randint(0, W - w, (1,)).item()
    mask[:, :, d_start:d_start + d, h_start:h_start + h, w_start:w_start + w] = 1
    return mask


def get_uniform_noise_mask(x):
    """
    Generate a uniform noise mask for the input tensor.

    Args:
        x (torch.tensor): Input tensor.

    Returns:
        torch.tensor: Uniform noise mask tensor.
    """
    mask = torch.rand_like(x)
    mask = (mask > (random.randint(200, 600) * 0.001)).float()
    return mask


def get_gaussian_noise_mask(x):
    """
    Generate a Gaussian noise mask for the input tensor.

    Args:
        x (torch.tensor): Input tensor of shape (B, C, D, H, W).

    Returns:
        torch.tensor: Gaussian noise mask tensor of the same shape as input.
    """
    B, C, D, H, W = x.shape
    mean = torch.randint(0, min(D, H, W) // 2, (1,)).item()
    sigma = torch.rand(1).item() * 10
    gauss = torch.randn(D, H, W) * sigma + mean
    gauss = (gauss - gauss.min()) / (gauss.max() - gauss.min())
    mask = torch.where(gauss > (random.randint(200, 600) * 0.001), torch.ones_like(x), torch.zeros_like(x))
    return mask


def get_reveen_mask(x):
    """
    Generate a Reveen mask for the input tensor.

    Args:
        x (torch.tensor): Input tensor of shape (B, C, D, H, W).

    Returns:
        torch.tensor: Reveen mask tensor of the same shape as input.
    """
    B, C, D, H, W = x.shape
    num_strips = torch.randint(5, 10, (1,))
    mask = torch.zeros_like(x)
    for _ in range(num_strips):
        d_start = torch.randint(0, D, (1,))
        h_start = torch.randint(0, H, (1,))
        w_start = torch.randint(0, W, (1,))
        thickness = torch.randint(5, 25, (1,))
        orientation = torch.randint(0, 3, (1,))  # 0 for D, 1 for H, 2 for W
        if orientation == 0:
            mask[:, :, d_start:d_start + thickness, :, :] = 1
        elif orientation == 1:
            mask[:, :, :, h_start:h_start + thickness, :] = 1
        elif orientation == 2:
            mask[:, :, :, :, w_start:w_start + thickness] = 1
    return mask


def get_rand_tissue_type_mask(x):
    """
    Generate a random tissue type mask for the input tensor.

    Args:
        x (torch.tensor): Input tensor.

    Returns:
        torch.tensor: Random tissue type mask tensor of the same shape as input.
    """
    contrast_range = torch.rand(2).sort().values
    if contrast_range[1] - contrast_range[0] > 0.35:
        if torch.rand(1).item() > 0.5:
            contrast_range[0] = contrast_range[1] - 0.35
        else:
            contrast_range[1] = contrast_range[0] + 0.35
    mask = torch.where((x > contrast_range[0]) & (x < contrast_range[1]), torch.ones_like(x), torch.zeros_like(x))
    return mask


def get_rand_blob_mask(x, blob_size_range=(8, 50)):
    """
    Generate a random blob mask for the input tensor.

    Args:
        x (torch.tensor): Input tensor.
        blob_size_range (tuple, optional): Range of blob sizes. Default is (8, 50).

    Returns:
        torch.tensor: Random blob mask tensor of the same shape as input.
    """
    mask = torch.zeros_like(x)
    num_blobs = random.randint(10, 35)

    for _ in range(num_blobs):
        blob_size = torch.randint(blob_size_range[0], blob_size_range[1], (1,)).item()
        start_indices = [torch.randint(0, max(1, dim - blob_size), (1,)).item() for dim in x.shape]
        blob_mask = torch.zeros_like(x)
        slices = tuple(slice(start, start + blob_size) for start in start_indices)
        blob_mask[slices] = 1
        mask = torch.max(mask, blob_mask)

    return mask


def get_blended_mask(x):
    """
    Generate a blended mask by combining different mask types for the input tensor.

    Args:
        x (torch.tensor): Input tensor.

    Returns:
        torch.tensor: Blended mask tensor of the same shape as input.
    """
    masks = [
        get_box_mask(x),
        get_uniform_noise_mask(x),
        get_gaussian_noise_mask(x),
        get_reveen_mask(x),
        get_rand_tissue_type_mask(x),
        get_rand_blob_mask(x)
    ]
    combined_mask = torch.stack(masks).sum(dim=0) / len(masks)
    return combined_mask >= (random.randint(2, 6) * 0.1)


def masking(x, mode='box'):
    """
    Apply masking to the input tensor based on the specified mode.

    Args:
        x (torch.tensor): Input tensor.
        mode (str, optional): Masking mode. Default is 'box'.

    Returns:
        torch.tensor: Masked tensor.
    """
    mask_functions = {
        'box': get_box_mask,
        'blob': get_rand_blob_mask,
        'noise_uniform': get_uniform_noise_mask,
        'noise_gaussian': get_gaussian_noise_mask,
        'reveen': get_reveen_mask,
        'tissue': get_rand_tissue_type_mask,
        'blended': get_blended_mask,
    }
    return mask_functions[mode](x).float()


def rand_augment(x):
    """
    Apply random augmentations to the input tensor.

    Args:
        x (torch.tensor): Input tensor.

    Returns:
        torch.tensor: Augmented tensor.
    """
    flip = tio.RandomFlip(axes=(0, 1, 2), p=0.5)
    affine = tio.RandomAffine(
        scales=(0.9, 1.1),
        degrees=180,
        translation=10,
        isotropic=False,
        center="image",
        image_interpolation='nearest'
    )
    transform = tio.Compose([flip, affine])
    transformed = transform(x)
    return transformed


def augment_batch(x):
    """
    Apply augmentation to a batch of input tensors.

    Args:
        x (torch.tensor): Batch of input tensors.

    Returns:
        torch.tensor: Augmented batch.
    """
    x = rand_augment(x[:, 0])[:, None, ...]
    return x


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
    image = torch.from_numpy(nib.load(os.path.join(path, os.path.join(filename, filename + '-t1n.nii.gz'))).get_fdata())[None, None, ...].float()
    if nrm:
        image = norm(image)
    return image


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
    def __init__(self, batch=1, augment=True, aug_thresh=0.05, workers=4, norm=True, path='/home/agam/Desktop/brats_2024_local_impainting/TrainingData/'):
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=workers)
        self.id = 0
        self.augment = augment
        self.aug_thresh = aug_thresh
        self.batch = batch
        self.path = path
        self.samples = list_folders(self.path)
        self.nrm = norm
        self.mask_modes = ['box', 'blob', 'noise_uniform', 'noise_gaussian', 'reveen', 'tissue', 'blended']
        self.randomize()
        self.future_batch = None
        self.pre_fetch_next_batch()

    def randomize(self):
        """
        Randomize the order of the samples.
        """
        sample_len = len(self.samples)
        self.max_id = sample_len - 1
        idx = random.sample(range(0, sample_len), sample_len)
        self.samples = [self.samples[i] for i in idx]

    def pre_fetch_next_batch(self):
        """
        Pre-fetch the next batch of samples asynchronously.
        """
        if self.id + self.batch > self.max_id:
            new_samples = self.samples[self.id:] if self.id < self.max_id else self.samples[self.id:self.id + 1]
            self.randomize()
            self.id = 0
        else:
            new_samples = self.samples[self.id:self.id + self.batch]
            self.id += self.batch

        self.future_batch = self.executor.submit(self.load_batch_dataset_async, new_samples)

    def load_batch_dataset_async(self, sample_list):
        """
        Load a batch of samples asynchronously.

        Args:
            sample_list (list): List of sample filenames.

        Returns:
            torch.tensor: Batch of loaded samples.
        """
        futures = [self.executor.submit(load_patient, self.path, sample, self.nrm) for sample in sample_list]
        results = [future.result() for future in futures]
        return torch.cat(results, dim=0)

    def load_batch(self):
        """
        Load a batch of samples and apply augmentation and masking.

        Returns:
            tuple: Batch of raw samples and corresponding masks.
        """
        batch_raw = self.future_batch.result()
        self.pre_fetch_next_batch()

        if self.augment and random.uniform(0, 1) > self.aug_thresh:
            if random.uniform(0, 1) > 0.3:
                batch_raw = torch.cat([slice_augment(batch_raw[:, i:i + 1], 20) for i in range(batch_raw.shape[1])], dim=1)
            batch_raw = augment_batch(batch_raw)

        batch_mask = masking(batch_raw, mode=self.mask_modes[random.randint(0, len(self.mask_modes) - 1)])
        return batch_raw, batch_mask


if __name__ == '__main__':
    from tqdm import trange
    loader = DataLoader(augment=True, batch=1, workers=4, path='/home/agam/Desktop/brats_2024_local_impainting/TrainingData/')
    for i in trange(1300):
        x, mask = loader.load_batch()
        x = x[0:2]
        mask = mask[0:2]
        x_ = x * (mask < 0.5)

        show_images(torch.cat((torch.permute(x, (0, 1, 4, 2, 3)), torch.permute(x, (0, 1, 4, 3, 2)), torch.permute(x_, (0, 1, 4, 2, 3)), torch.permute(x_, (0, 1, 4, 3, 2))), dim=0), 8, 4, dpi=250)
