import cv2
import numpy as np
from typing import List
from PIL import Image, ImageChops

import torch
import torch.nn.functional as F

def concat_images_row(images: List[Image.Image], bg_color=(0, 0, 0)) -> Image.Image:
    """
    Concatenate multiple PIL images horizontally (side by side) into a single image.
    
    Args:
        images: List of images to concatenate.
        bg_color: Background color, default black; use (0,0,0,0) if images have an alpha channel.
    
    Returns:
        A new image with all inputs concatenated horizontally.
    """
    if not images:
        raise ValueError("images list must not be empty")
    
    # Unify mode: use RGBA if any image has an alpha channel, otherwise RGB
    modes = {img.mode for img in images}
    mode = "RGBA" if any(m.endswith("A") for m in modes) else "RGB"

    # Compute canvas dimensions
    widths, heights = zip(*(img.size for img in images))
    total_width = sum(widths)
    max_height = max(heights)

    # Create canvas
    canvas = Image.new(mode, (total_width, max_height), bg_color)

    # Paste images one by one
    x_offset = 0
    for img in images:
        # Convert mode if necessary
        if img.mode != mode:
            img = img.convert(mode)
        canvas.paste(img, (x_offset, 0), img if mode=="RGBA" else None)
        x_offset += img.width
    return canvas

def downsample_mask_pytorch(pil_mask: Image.Image, factor: int) -> Image.Image:
    """
    Downsample a binary mask using PyTorch's max_pool2d, preserving any white pixel within each block.
    
    Args:
        pil_mask: A binary PIL image in mode '1' or 'L' (values 0/255).
        factor: Downsampling factor (used as both kernel size and stride).
    
    Returns:
        Downsampled binary PIL Image (mode='1').
    """
    # Convert to 0/1 float tensor with shape [1,1,H,W]
    arr = np.array(pil_mask.convert('L'), dtype=np.uint8)
    tensor = torch.from_numpy(arr).float().div_(255.0).unsqueeze(0).unsqueeze(0)
    
    # Downsample with max_pool2d
    pooled = F.max_pool2d(tensor, kernel_size=factor, stride=factor)
    
    # Recover to 0/255 and convert back to PIL
    out = (pooled.squeeze(0).squeeze(0) > 0).to(torch.uint8).mul_(255).cpu().numpy()
    return Image.fromarray(out, mode='L').convert('1')

def create_all_white_like(pil_img: Image.Image) -> Image.Image:
    """
    Given a PIL image, return an all-white binary image (mode='1') of the same size.
    """
    w, h = pil_img.size
    white_array = np.ones((h, w), dtype=np.uint8) * 255  # note: shape is (H, W)
    return Image.fromarray(white_array, mode='L').convert('1')

def union_masks_np(masks: List[Image.Image]) -> Image.Image:
    """
    Accept a list of PIL Images (binary, mode='1' or 'L') and return their union.
    """
    if not masks:
        raise ValueError("Input masks list must not be empty")

    # Convert each image to a 0/1 numpy array
    bin_arrays = []
    for m in masks:
        arr = np.array(m.convert('L'), dtype=np.uint8)
        bin_arr = (arr > 127).astype(np.bool_)
        bin_arrays.append(bin_arr)

    # Pixel-wise logical OR
    union_bool = np.logical_or.reduce(bin_arrays)

    # Recover to 0/255 uint8
    union_arr = union_bool.astype(np.uint8) * 255

    # Convert back to PIL (binary)
    return Image.fromarray(union_arr, mode='L').convert('1')

def intersect_masks_np(masks: List[Image.Image]) -> Image.Image:
    """
    Accept a list of PIL Images (binary, mode='1' or 'L') and return their intersection.
    """
    if not masks:
        raise ValueError("Input masks list must not be empty")

    # Convert each image to a 0/1 numpy array
    bin_arrays = []
    for m in masks:
        arr = np.array(m.convert('L'), dtype=np.uint8)
        bin_arr = (arr > 127).astype(np.bool_)
        bin_arrays.append(bin_arr)

    # Pixel-wise logical AND
    intersect_bool = np.logical_and.reduce(bin_arrays)

    # Recover to 0/255 uint8
    intersect_arr = intersect_bool.astype(np.uint8) * 255

    # Convert back to PIL (binary)
    return Image.fromarray(intersect_arr, mode='L').convert('1')

def close_small_holes(pil_mask, kernel_size=5):
    """
    Fill small black holes using morphological closing.
    kernel_size: structuring element size; larger values fill bigger holes. Typically an odd number.
    """
    # 1. Convert to 0/255 binary
    mask = np.array(pil_mask.convert('L'))
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Define structuring element
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    # 3. Morphological closing
    closed = cv2.morphologyEx(bin_mask, cv2.MORPH_CLOSE, kernel)
    return Image.fromarray(closed)

def get_mask(src_image, tgt_image, threshold=1):
    """
    Pixels with large differences (large diff value) are treated as foreground (white),
    otherwise background (black).
    """
    diff = ImageChops.difference(src_image, tgt_image)
    diff_gray = diff.convert("L")
    mask = diff_gray.point(lambda x: 255 if x >= threshold else 0).convert("1")
    return mask

def filter_small_components(pil_mask, area_threshold=0.10):
    """
    Remove connected white regions smaller than area_threshold (default 10%) of total image area.
    pil_mask: PIL.Image in mode 'L' or '1' (0/255 binary image).
    area_threshold: threshold as a fraction of the total image area.
    Returns: processed PIL.Image.
    """
    # 1. Convert to binary NumPy array (0, 255)
    mask = np.array(pil_mask.convert('L'))
    # Ensure values are 0/255
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # 2. Label connected components (4- or 8-connectivity both work)
    num_labels, labels = cv2.connectedComponents(bin_mask, connectivity=8)
    h, w = bin_mask.shape
    total_area = h * w
    total_area = np.count_nonzero(bin_mask)
    
    # 3. Iterate over each connected component
    output = np.zeros_like(bin_mask)
    for lbl in range(1, num_labels):  # 0 is background
        # Extract this component
        comp_mask = (labels == lbl)
        comp_area = comp_mask.sum()
        # Area ratio
        if comp_area >= area_threshold * total_area:
            # Keep it
            output[comp_mask] = 255
    
    # 4. Convert back to PIL
    return Image.fromarray(output)

def is_binary_255(t: torch.Tensor) -> bool:
    """
    Check whether the given tensor contains only the values 0 and 255.
    """
    unique_vals = torch.unique(t)
    return torch.equal(unique_vals, torch.tensor([0], dtype=t.dtype)) or \
           torch.equal(unique_vals, torch.tensor([255], dtype=t.dtype)) or \
           torch.equal(unique_vals, torch.tensor([0, 255], dtype=t.dtype))

def get_weight(mask_u_ds, weight_type='log'):
    mask_u_ds_tensor = torch.from_numpy(np.array(mask_u_ds)).float()
    assert is_binary_255(mask_u_ds_tensor), "is_binary_255(mask_u_ds_tensor)"
    mask_u_ds_tensor_bool = mask_u_ds_tensor.bool()
    x = mask_u_ds_tensor_bool.numel() / mask_u_ds_tensor_bool.sum()
    if weight_type == 'log':
        weight = torch.log2(x) + 1
    elif weight_type == 'exp':
        weight = 2 ** (x**0.5 - 1)
    else:
        raise NotImplementedError(f'Support log | exp, but found {weight_type}')
    weight = torch.round(weight, decimals=6)
    assert weight >= 1, \
        f"weight >= 1 but {weight}, {mask_u_ds_tensor_bool.shape}, mask_u_ds_tensor_bool.numel(): {mask_u_ds_tensor_bool.numel()}, mask_u_ds_tensor_bool.sum(): {mask_u_ds_tensor_bool.sum()}"
    mask_u_ds_tensor[mask_u_ds_tensor==255] = weight
    mask_u_ds_tensor[mask_u_ds_tensor==0] = 1.0
    return mask_u_ds_tensor.unsqueeze(0)  # h w -> 1 h w

def get_weight_mask(pil_pixel_values, prompt=None, weight_type='log', need_weight='true'):
    # area_threshold = 1/64
    area_threshold = 0.001
    # base_kernel_size_factor = (5 / 448) ** 2
    # if len(pil_pixel_values) > 0:
    #     w, h = pil_pixel_values[-1].size
    #     kernel_size = max(int((base_kernel_size_factor * h * w) ** 0.5), 3)
    # else:
    kernel_size = 5
    if need_weight.lower() == 'false':
        mask_intersect = create_all_white_like(pil_pixel_values[-1])
        mask_intersect_ds = downsample_mask_pytorch(mask_intersect, factor=8)  # factor is the VAE downsampling ratio
        mask_intersect_ds = close_small_holes(mask_intersect_ds, kernel_size=kernel_size)
        weight = get_weight(mask_intersect_ds, weight_type)
        return mask_intersect_ds, weight
    filtered_masks = []
    for ii, j in enumerate(pil_pixel_values[:-1]):
        # compare each reference image against the target image to obtain a difference mask
        mask = get_mask(j, pil_pixel_values[-1], threshold=18)
        # fill small holes
        fill_mask = close_small_holes(mask, kernel_size=kernel_size)
        # remove small connected components
        filtered_mask = filter_small_components(fill_mask, area_threshold=0.3)
        # filtered_mask = fill_mask
        filtered_masks.append(filtered_mask)
    if len(filtered_masks) == 0:
        # t2i tasks have no reference image
        assert len(pil_pixel_values) == 1, "len(pil_pixel_values) == 1"
        mask_intersect = create_all_white_like(pil_pixel_values[-1])
    else:
        mask_intersect = intersect_masks_np(filtered_masks)
    # the foreground area ratio must exceed 1/16 (a minimum threshold)
    mask_intersect_area_ratio = np.array(mask_intersect).astype(np.float32).sum() / np.prod(np.array(mask_intersect).shape)
    # print(mask_intersect_area_ratio)
    if mask_intersect_area_ratio < area_threshold:
        if mask_intersect_area_ratio == 0.0:
            # ratio == 0 indicates reconstruction data from stage 1
            assert len(pil_pixel_values) == 2, "len(pil_pixel_values) == 2"
            mask_intersect = create_all_white_like(pil_pixel_values[-1])
        else:
            # concat_images_row(pil_pixel_values + [mask_intersect], bg_color=(255,255,255)).show()
            raise ValueError(f'TOO SMALL mask_intersect_area_ratio: {mask_intersect_area_ratio}, prompt: {prompt}')
    mask_intersect_ds = downsample_mask_pytorch(mask_intersect, factor=8)  # factor is the VAE downsampling ratio
    mask_intersect_ds = close_small_holes(mask_intersect_ds, kernel_size=kernel_size)
    weight = get_weight(mask_intersect_ds, weight_type)
    return mask_intersect_ds, weight

def get_weight_mask_test(pil_pixel_values, prompt=None, weight_type='log'):
    area_threshold = 1/64
    base_kernel_size_factor = (5 / 448) ** 2
    if len(pil_pixel_values) > 0:
        w, h = pil_pixel_values[-1].size
        kernel_size = max(int((base_kernel_size_factor * h * w) ** 0.5), 3)
    else:
        kernel_size = 5

    filtered_masks = []
    for ii, j in enumerate(pil_pixel_values[:-1]):
        # compare each reference image against the target image to obtain a difference mask
        mask = get_mask(j, pil_pixel_values[-1], threshold=18)
        # fill small holes
        fill_mask = close_small_holes(mask, kernel_size=kernel_size)
        # remove small connected components
        filtered_mask = filter_small_components(fill_mask, area_threshold=1/64)
        # filtered_mask = fill_mask
        filtered_masks.append(filtered_mask)
    if len(filtered_masks) == 0:
        # t2i tasks have no reference image
        assert len(pil_pixel_values) == 1, "len(pil_pixel_values) == 1"
        mask_intersect = create_all_white_like(pil_pixel_values[-1])
    else:
        mask_intersect = intersect_masks_np(filtered_masks)
    # the foreground area ratio must exceed 1/16 (a minimum threshold)
    mask_intersect_area_ratio = np.array(mask_intersect).astype(np.float32).sum() / np.prod(np.array(mask_intersect).shape)
    return mask_intersect_area_ratio