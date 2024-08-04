import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms


def load_image_as_tensor_greyscale(image_path):
    """
    Load a .png image as a grayscale PyTorch tensor with shape (1, 1, 64, 64).
    Args:
        image_path (str): The path to the .png image file.
    Returns:
        torch.Tensor: The grayscale image as a tensor with shape (1, 1, 64, 64).
    """
    # Open the image using PIL and convert it to grayscale
    image = Image.open(image_path).convert("L")

    # Define a transform to resize and convert the image to a tensor
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Apply the transform to the image
    image_tensor = transform(image)

    # Add an extra dimension to match the desired shape (1, 1, 64, 64)
    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor


def match_histogram(source, reference, mask):
    """
    Match the histogram of the source image to that of the reference image.
    Args:
        source (torch.Tensor): Source image tensor of shape (1, 1, ...)
        reference (torch.Tensor): Reference image tensor of shape (1, 1, ...)
        mask (torch.Tensor): Mask for inpainted region (1, 1, ...)
    Returns:
        torch.Tensor: Transformed source image tensor with matched histogram.
    """
    source_mask = (torch.where(reference > 0, 1, 0) + mask)
    reference_mask = torch.where(reference > 0, 1, 0)

    # Extract masked values from source and reference images
    source_values = source[source_mask.bool()]
    reference_values = reference[reference_mask.bool()]

    # Get unique values and their counts from the masked regions
    src_vals, src_counts = torch.unique(source_values, return_counts=True)
    ref_vals, ref_counts = torch.unique(reference_values, return_counts=True)

    # Compute the cumulative distribution function (CDF)
    src_cdf = torch.cumsum(src_counts.float(), dim=0)
    ref_cdf = torch.cumsum(ref_counts.float(), dim=0)

    # Normalize the CDFs
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]

    # Convert to numpy for interpolation
    src_cdf_np = src_cdf.numpy()
    ref_cdf_np = ref_cdf.numpy()
    ref_vals_np = ref_vals.numpy()

    # Create a mapping function from source to reference using numpy.interp
    interp_tgt_vals = np.interp(src_cdf_np, ref_cdf_np, ref_vals_np)

    # Map the source values to the new values
    matched_values = torch.zeros_like(source_values, dtype=torch.float32)
    for i, val in enumerate(src_vals):
        matched_values[source_values == val] = torch.tensor(
            interp_tgt_vals[i], dtype=torch.float32)

    # Create a copy of the source image to store the result
    result_image = source.clone()

    # Insert the matched values back into the result image using the source mask
    result_image[source_mask.bool()] = matched_values

    return result_image.to(device=reference.device)


# Good Example. For global correction:
image_path = "/home/abc.png"  # Replace with your image path
image_tensor = load_image_as_tensor_greyscale(image_path)
mask = torch.zeros((1, 1, 64, 64))  # Masked region
mask[:, :, 20:48, 20:48] += 1
image1 = image_tensor * torch.where(mask < 0.5, 1, 0)  # Reference image
image_path = "/home/xyz.png"  # Replace with your image path
image_tensor = load_image_as_tensor_greyscale(
    image_path) * torch.nn.functional.sigmoid(torch.rand_like(mask))/2
image2 = image_tensor  # Source image to match

matched_image = match_histogram(image2, image1, mask)

mask[:, :, 0, 0], mask[:, :, 0, 1] = 0, 1
image1[:, :, 0, 0], image1[:, :, 0, 1] = 0, 1
image2[:, :, 0, 0], image2[:, :, 0, 1] = 0, 1
matched_image[:, :, 0, 0], matched_image[:, :, 0, 1] = 0, 1

plt.imshow(mask[0, 0], cmap='grey')
plt.show()
plt.imshow(image1[0, 0], cmap='grey')
plt.show()
plt.imshow(image2[0, 0], cmap='grey')
plt.show()
plt.imshow(matched_image[0, 0], cmap='grey')
plt.show()


# Bad Example. For local correction:
image_path = "/home/abc.png"  # Replace with your image path
image_tensor = load_image_as_tensor_greyscale(image_path)
mask = torch.zeros((1, 1, 64, 64))  # Masked region
mask[:, :, 20:48, 20:48] += 1
image1 = image_tensor * torch.where(mask < 0.5, 1, 0)  # Reference image
image_path = "/home/xyz.png"  # Replace with your image path
image_tensor = load_image_as_tensor_greyscale(image_path)
image2 = image_tensor  # Source image to match
image2[:, :, 20:48, 20:48] *= torch.nn.functional.sigmoid(
    torch.rand_like(image2[:, :, 20:48, 20:48]))/2

matched_image = match_histogram(image2, image1, mask)

mask[:, :, 0, 0], mask[:, :, 0, 1] = 0, 1
image1[:, :, 0, 0], image1[:, :, 0, 1] = 0, 1
image2[:, :, 0, 0], image2[:, :, 0, 1] = 0, 1
matched_image[:, :, 0, 0], matched_image[:, :, 0, 1] = 0, 1

plt.imshow(mask[0, 0], cmap='grey')
plt.show()
plt.imshow(image1[0, 0], cmap='grey')
plt.show()
plt.imshow(image2[0, 0], cmap='grey')
plt.show()
plt.imshow(matched_image[0, 0], cmap='grey')
plt.show()


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


# Example usage
image_path = "/home/abc.png"  # Replace with your image path
image_tensor = load_image_as_tensor_greyscale(image_path)
mask = torch.zeros((1, 1, 64, 64))  # Masked region
mask[:, :, 20:48, 20:48] += 1
image1 = image_tensor * torch.where(mask < 0.5, 1, 0)  # Reference image
image_path = "/home/xyz.png"  # Replace with your image path
image2 = 0.3 * load_image_as_tensor_greyscale(
    image_path) * mask * torch.nn.functional.sigmoid(torch.rand_like(mask))

image1[:, :, 0, 0], image1[:, :, 0, 1] = 0, 1
image2[:, :, 0, 0], image2[:, :, 0, 1] = 0, 1
plt.imshow(image1[0, 0], cmap='grey')
plt.show()
plt.imshow(image2[0, 0], cmap='grey')
plt.show()
plt.imshow(mask[0, 0], cmap='grey')
plt.show()

matched_image, surrounding_mask = match_contrast_inpaint(image2, image1, mask)
matched_image += image1

matched_image[:, :, 0, 0], matched_image[:, :, 0, 1] = 0, 1

plt.imshow(surrounding_mask[0, 0], cmap='grey')
plt.show()
plt.imshow(matched_image[0, 0], cmap='grey')
plt.show()
