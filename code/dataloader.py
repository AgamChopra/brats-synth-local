"""
Created on June 2023, Modified on May 2024,
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

from utils import norm, show_images  # , pad3d


def slice_augment(x, min_slices=12):
    B, C, D, H, W = x.shape
    target_depth = torch.randint(min_slices, D + 1, (1,)).item()
    target_height = torch.randint(min_slices, H + 1, (1,)).item()
    target_width = torch.randint(min_slices, W + 1, (1,)).item()

    downsampled = F.interpolate(x, size=(
        target_depth, target_height, target_width),
        mode='trilinear', align_corners=False)

    augmented = F.interpolate(downsampled, size=(
        D, H, W), mode='nearest')

    # if random.uniform(0, 1) > 0.5:
    #     augmented = F.interpolate(downsampled, size=(
    #         D, H, W), mode='nearest')
    # else:
    #     augmented = pad3d(downsampled, (D, H, W))

    return augmented


def get_box_mask(x):
    B, C, D, H, W = x.shape
    d = torch.randint(D//4, D//2, (1,)).item()
    h = torch.randint(H//4, H//2, (1,)).item()
    w = torch.randint(W//4, W//2, (1,)).item()
    mask = torch.zeros_like(x)
    d_start = torch.randint(0, D - d, (1,)).item()
    h_start = torch.randint(0, H - h, (1,)).item()
    w_start = torch.randint(0, W - w, (1,)).item()
    mask[:, :, d_start:d_start + d, h_start:h_start + h,
         w_start:w_start + w] = 1
    return mask


def get_uniform_noise_mask(x):
    mask = torch.rand_like(x)
    mask = (mask > (random.randint(200, 600) * 0.001)).float()
    return mask


def get_gaussian_noise_mask(x):
    B, C, D, H, W = x.shape
    mean = torch.randint(0, min(D, H, W)//2, (1,)).item()
    sigma = torch.rand(1).item() * 10
    gauss = torch.randn(D, H, W) * sigma + mean
    gauss = (gauss - gauss.min()) / (gauss.max() - gauss.min())
    mask = torch.where(gauss > (random.randint(200, 600) * 0.001),
                       torch.ones_like(x), torch.zeros_like(x))
    return mask


def get_reveen_mask(x):
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
            mask[:, :, d_start:d_start+thickness, :, :] = 1
        elif orientation == 1:
            mask[:, :, :, h_start:h_start+thickness, :] = 1
        elif orientation == 2:
            mask[:, :, :, :, w_start:w_start+thickness] = 1
    return mask


def get_rand_tissue_type_mask(x):
    contrast_range = torch.rand(2).sort().values
    if contrast_range[1] - contrast_range[0] > 0.35:
        if torch.rand(1).item() > 0.5:
            contrast_range[0] = contrast_range[1] - 0.35
        else:
            contrast_range[1] = contrast_range[0] + 0.35
    mask = torch.where((x > contrast_range[0]) & (
        x < contrast_range[1]), torch.ones_like(x), torch.zeros_like(x))
    return mask


def get_rand_blob_mask(x, blob_size_range=(8, 50)):
    mask = torch.zeros_like(x)
    num_blobs = random.randint(10, 35)

    for _ in range(num_blobs):
        blob_size = torch.randint(
            blob_size_range[0], blob_size_range[1], (1,)).item()

        start_indices = [torch.randint(
            0, max(1, dim - blob_size), (1,)).item() for dim in x.shape]

        blob_mask = torch.zeros_like(x)
        slices = tuple(slice(start, start + blob_size)
                       for start in start_indices)
        blob_mask[slices] = 1
        mask = torch.max(mask, blob_mask)

    return mask


def get_blended_mask(x):
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
    x = rand_augment(x[:, 0])[:, None, ...]
    return x


def load_patient(path, filename, nrm=True):
    image = torch.from_numpy(nib.load(os.path.join(
        path, os.path.join(filename, filename + '-t1n.nii.gz')
    )).get_fdata())[None, None, ...].float()
    if nrm:
        image = norm(image)
    return image


def list_folders(path):
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


class DataLoader():
    def __init__(self, batch=1, augment=True,
                 aug_thresh=0.05, workers=4, norm=True,
                 path='/home/agam/Desktop/brats_2024_local_impainting/TrainingData/'):
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=workers)
        self.id = 0
        self.augment = augment
        self.aug_thresh = aug_thresh
        self.batch = batch
        self.path = path
        self.samples = list_folders(self.path)
        self.nrm = norm
        self.mask_modes = ['box', 'blob', 'noise_uniform',
                           'noise_gaussian', 'reveen', 'tissue', 'blended']
        self.randomize()
        self.future_batch = None
        self.pre_fetch_next_batch()

    def randomize(self):
        sample_len = len(self.samples)
        self.max_id = sample_len - 1
        idx = random.sample(range(0, sample_len), sample_len)
        self.samples = [self.samples[i] for i in idx]

    def pre_fetch_next_batch(self):
        if self.id + self.batch > self.max_id:
            new_samples = self.samples[
                self.id:] if self.id < self.max_id else self.samples[
                    self.id:self.id+1]
            self.randomize()
            self.id = 0
        else:
            new_samples = self.samples[self.id:self.id + self.batch]
            self.id += self.batch

        self.future_batch = self.executor.submit(
            self.load_batch_dataset_async, new_samples)

    def load_batch_dataset_async(self, sample_list):
        futures = []
        for sample in sample_list:
            future = self.executor.submit(
                load_patient, self.path, sample, self.nrm)
            futures.append(future)
        results = [future.result() for future in futures]
        return torch.cat(results, dim=0)

    def load_batch(self):
        batch_raw = self.future_batch.result()

        self.pre_fetch_next_batch()

        if self.augment and random.uniform(0, 1) > self.aug_thresh:
            if random.uniform(0, 1) > 0.3:
                batch_raw = torch.cat([slice_augment(batch_raw[:, i:i+1], 20)
                                       for i in range(batch_raw.shape[1])],
                                      dim=1)
            batch_raw = augment_batch(batch_raw)

        batch_mask = masking(
            batch_raw, mode=self.mask_modes[
                random.randint(0, len(self.mask_modes) - 1)])

        return batch_raw, batch_mask


if __name__ == '__main__':
    from tqdm import trange
    loader = DataLoader(augment=True,
                        batch=1, workers=4,
                        path='/home/agam/Desktop/brats_2024_local_impainting/TrainingData/')
    for i in trange(1300):
        x, mask = loader.load_batch()
        x = x[0:2]
        mask = mask[0:2]
        x_ = x * (mask < 0.5)

        show_images(torch.cat((torch.permute(x, (0, 1, 4, 2, 3)),
                    torch.permute(x, (0, 1, 4, 3, 2)),
                    torch.permute(x_, (0, 1, 4, 2, 3)),
                    torch.permute(x_, (0, 1, 4, 3, 2))), dim=0), 8, 4, dpi=250)
