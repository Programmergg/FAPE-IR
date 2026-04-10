import math
from PIL import Image

RATIO = {
    'any_11ratio': [(16, 9), (9, 16), (7, 5), (5, 7), (5, 4), (4, 5), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_9ratio': [(16, 9), (9, 16), (5, 4), (4, 5), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_7ratio': [(16, 9), (9, 16), (4, 3), (3, 4), (3, 2), (2, 3), (1, 1)],
    'any_5ratio': [(16, 9), (9, 16), (4, 3), (3, 4), (1, 1)],
    'any_1ratio': [(1, 1)],
}

def dynamic_resize(h, w, anyres='any_1ratio', anchor_pixels=1024 * 1024, stride=32):
    
    orig_ratio = w / h

    # Find the candidate aspect ratio closest to the original image ratio
    target_ratio = min(RATIO[anyres], key=lambda x: abs((x[0] / x[1]) - orig_ratio))
    rw, rh = target_ratio

    # Compute the minimum baseline dimensions aligned to stride
    base_h = rh * stride
    base_w = rw * stride
    base_area = base_h * base_w

    # Compute the scale factor that brings the area close to anchor_pixels
    # under the chosen aspect ratio and stride alignment
    scale = round(math.sqrt(anchor_pixels / base_area))

    new_h = base_h * scale
    new_w = base_w * scale

    return new_h, new_w


def concat_images_adaptive(images, bg_color=(255, 255, 255)):
    """
    Adaptively tile an arbitrary number of PIL.Image objects into an
    approximately square grid image.

    Args:
        images (list of PIL.Image): List of images to concatenate.
        bg_color (tuple of int): Background color, default white (255, 255, 255).

    Returns:
        PIL.Image: The resulting tiled image.
    """
    if not images:
        raise ValueError("images list must not be empty")

    n = len(images)

    # Compute grid dimensions (rows × cols) as close to square as possible
    cols = int(n**0.5)
    if cols * cols < n:
        cols += 1
    rows = (n + cols - 1) // cols

    # Find the maximum width and height across all images
    widths, heights = zip(*(img.size for img in images))
    max_w = max(widths)
    max_h = max(heights)

    # Create the canvas
    new_img = Image.new('RGB', (cols * max_w, rows * max_h), color=bg_color)

    # Paste images row by row, column by column; empty cells remain blank
    for idx, img in enumerate(images):
        row_idx = idx // cols
        col_idx = idx % cols
        # If image sizes differ, center each image within its cell
        offset_x = col_idx * max_w + (max_w - img.width) // 2
        offset_y = row_idx * max_h + (max_h - img.height) // 2
        new_img.paste(img, (offset_x, offset_y))

    return new_img