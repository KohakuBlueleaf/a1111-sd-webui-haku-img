from functools import partial
from itertools import product
from PIL import Image
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Union


def wavelet_blur(inp, radius):
    kernel_size = 2 * radius + 1
    output = cv2.GaussianBlur(inp, (kernel_size, kernel_size), 0)
    return output


def wavelet_decomposition(inp, levels):
    high_freq = np.zeros_like(inp)
    for i in range(1, levels + 1):
        radius = 2**i
        low_freq = wavelet_blur(inp, radius)
        high_freq = high_freq + (inp - low_freq)
        inp = low_freq
    return high_freq, low_freq


def wavelet_colorfix(inp, target):
    inp_high, _ = wavelet_decomposition(inp, 5)
    _, target_low = wavelet_decomposition(target, 5)
    output = inp_high + target_low
    return output


def match_color(source, target):
    # Convert RGB to L*a*b*, and then match the std/mean
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32) / 255
    result = (source_lab - np.mean(source_lab)) / np.std(source_lab)
    result = result * np.std(target_lab) + np.mean(target_lab)
    source = cv2.cvtColor(
        (result * 255).clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR
    )

    source = source.astype(np.float32)
    # Use wavelet colorfix method to match original low frequency data at first
    source[:, :, 0] = wavelet_colorfix(source[:, :, 0], target[:, :, 0])
    source[:, :, 1] = wavelet_colorfix(source[:, :, 1], target[:, :, 1])
    source[:, :, 2] = wavelet_colorfix(source[:, :, 2], target[:, :, 2])
    output = source
    return output.clip(0, 255).astype(np.uint8)


def color_styling(inp, saturation=1.2, contrast=1.1):
    output = inp.copy()
    output = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    output[:, :, 1] = output[:, :, 1] * saturation
    output[:, :, 2] = output[:, :, 2] * contrast - (contrast - 1)
    output = np.clip(output, 0, 1)
    output = cv2.cvtColor(output, cv2.COLOR_HSV2BGR)
    return output


def kmeans_color_quant(inp, colors=32):
    img = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)
    img_quant = img_pil.quantize(colors, 1, kmeans=colors).convert("RGB")
    return cv2.cvtColor(np.array(img_quant), cv2.COLOR_RGB2BGR)


def bicubic(
    img,
    target_size=128,
):
    H, W, _ = img.shape

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img_sm = img.resize(target_hw, Image.BICUBIC)
    img_sm = cv2.cvtColor(np.asarray(img_sm), cv2.COLOR_RGB2BGR)
    return img_sm


def nearest(
    img,
    target_size=128,
):
    H, W, _ = img.shape

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img_rgb)
    img_sm = img.resize(target_hw, Image.NEAREST)
    img_sm = cv2.cvtColor(np.asarray(img_sm), cv2.COLOR_RGB2BGR)
    return img_sm


@torch.no_grad()
def apply_chunk(data, kernel, stride, func):
    org_shape = data.shape
    unfold_shape = org_shape

    k_shift = max(kernel - stride, 0)
    pad_pattern = (k_shift // 2, k_shift // 2 + k_shift % 2)
    data = np.pad(data, (pad_pattern, pad_pattern), "edge")

    if len(org_shape) == 2:
        data = data[np.newaxis, np.newaxis, ...]

    data = (
        F.unfold(torch.tensor(data), kernel, 1, 0, stride).transpose(-1, -2)[0].numpy()
    )
    data[..., : stride**2] = func(data)
    data = data[np.newaxis, ..., : stride**2]
    data = F.fold(
        torch.tensor(data).transpose(-1, -2),
        unfold_shape,
        stride,
        1,
        0,
        stride,
    )[0].numpy()

    if len(org_shape) < 3:
        data = data[0]
    return data


def center_downscale(
    img,
    target_size=128,
):
    H, W, _ = img.shape

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    patch_size = max(int(round(H // target_hw[1])), int(round(W // target_hw[0])))

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab[:, :, 0] = apply_chunk(
        img_lab[:, :, 0],
        patch_size,
        patch_size,
        lambda x: x[..., x.shape[-1] // 2][..., None],
    )
    img_lab[:, :, 1] = apply_chunk(
        img_lab[:, :, 1],
        patch_size,
        patch_size,
        partial(np.median, axis=1, keepdims=True),
    )
    img_lab[:, :, 2] = apply_chunk(
        img_lab[:, :, 2],
        patch_size,
        patch_size,
        partial(np.median, axis=1, keepdims=True),
    )
    img = cv2.cvtColor(img_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    img_sm = cv2.resize(img, target_hw, interpolation=cv2.INTER_NEAREST)
    return img_sm


def find_pixel(chunks):
    mid = chunks[..., chunks.shape[-1] // 2][..., np.newaxis]
    med = np.median(chunks, axis=1, keepdims=True)
    mu = np.mean(chunks, axis=1, keepdims=True)
    maxi = np.max(chunks, axis=1, keepdims=True)
    mini = np.min(chunks, axis=1, keepdims=True)

    output = mid
    mini_loc = (med < mu) & (maxi - med > med - mini)
    maxi_loc = (med > mu) & (maxi - med < med - mini)

    output[mini_loc] = mini[mini_loc]
    output[maxi_loc] = maxi[maxi_loc]

    return output


def contrast_based_downscale(
    img,
    target_size=128,
):
    H, W, _ = img.shape

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    target_hw = (int(target_size * ratio), int(target_size))
    patch_size = max(int(round(H // target_hw[1])), int(round(W // target_hw[0])))

    img_lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    img_lab[:, :, 0] = apply_chunk(img_lab[:, :, 0], patch_size, patch_size, find_pixel)
    img_lab[:, :, 1] = apply_chunk(
        img_lab[:, :, 1],
        patch_size,
        patch_size,
        partial(np.median, axis=1, keepdims=True),
    )
    img_lab[:, :, 2] = apply_chunk(
        img_lab[:, :, 2],
        patch_size,
        patch_size,
        partial(np.median, axis=1, keepdims=True),
    )
    img = cv2.cvtColor(img_lab.clip(0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

    img_sm = cv2.resize(img, target_hw, interpolation=cv2.INTER_NEAREST)
    return img_sm


def k_centroid_downscale(cv2img, target_size=128, centroids=2):
    """
    k-centroid downscaling algorithm from Astropulse, under MIT License.
    https://github.com/Astropulse/pixeldetector/blob/6e88e18ddbd16529b5dd85b1c615cbb2e5778bf2/k-centroid.py#L19-L44
    """
    H, W, _ = cv2img.shape

    ratio = W / H
    target_size = (target_size**2 / ratio) ** 0.5
    height = int(target_size)
    width = int(target_size * ratio)

    # Perform outline expansion and color matching
    image = Image.fromarray(cv2.cvtColor(cv2img, cv2.COLOR_BGR2RGB)).convert("RGB")

    # Downscale outline expanded image with k-centroid
    # Create an empty array for the downscaled image
    downscaled = np.zeros((height, width, 3), dtype=np.uint8)

    # Calculate the scaling factors
    wFactor = image.width / width
    hFactor = image.height / height

    # Iterate over each tile in the downscaled image
    for x, y in product(range(width), range(height)):
        # Crop the tile from the original image
        tile = image.crop(
            (x * wFactor, y * hFactor, (x * wFactor) + wFactor, (y * hFactor) + hFactor)
        )

        # Quantize the colors of the tile using k-means clustering
        tile = tile.quantize(colors=centroids, method=1, kmeans=centroids).convert(
            "RGB"
        )

        # Get the color counts and find the most common color
        color_counts = tile.getcolors()
        most_common_color = max(color_counts, key=lambda x: x[0])[1]

        # Assign the most common color to the corresponding pixel in the downscaled image
        downscaled[y, x, :] = most_common_color

    return cv2.cvtColor(downscaled, cv2.COLOR_RGB2BGR)


downscale_mode = {
    "bicubic": bicubic,
    "nearest": nearest,
    "center": center_downscale,
    "contrast": contrast_based_downscale,
    "k-centroid": k_centroid_downscale,
}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def expansion_weight(img, k=8, stride=2, avg_scale=10, dist_scale=3):
    img_y = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[:, :, 0] / 255
    avg_y = apply_chunk(img_y, k * 2, stride, partial(np.median, axis=1, keepdims=True))
    max_y = apply_chunk(img_y, k, stride, partial(np.max, axis=1, keepdims=True))
    min_y = apply_chunk(img_y, k, stride, partial(np.min, axis=1, keepdims=True))
    bright_dist = max_y - avg_y
    dark_dist = avg_y - min_y

    weight = (avg_y - 0.5) * avg_scale
    weight = weight - (bright_dist - dark_dist) * dist_scale

    output = sigmoid(weight)
    output = cv2.resize(
        output,
        (img.shape[1] // stride, img.shape[0] // stride),
        interpolation=cv2.INTER_LINEAR,
    )
    output = cv2.resize(
        output, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR
    )

    return (output - np.min(output)) / (np.max(output))


def outline_expansion(img, erode=2, dilate=2, k=16, avg_scale=10, dist_scale=3):
    weight = expansion_weight(img, k, (k // 4) * 2, avg_scale, dist_scale)[
        ..., np.newaxis
    ]
    orig_weight = sigmoid((weight - 0.5) * 5) * 0.25
    kernel_expansion = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.uint8)
    kernel_smoothing = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.uint8)

    img_erode = img.copy()
    img_erode = cv2.erode(img_erode, kernel_expansion, iterations=erode).astype(
        np.float32
    )
    img_dilate = img.copy()
    img_dilate = cv2.dilate(img_dilate, kernel_expansion, iterations=dilate).astype(
        np.float32
    )

    output = img_erode * weight + img_dilate * (1 - weight)
    output = output * (1 - orig_weight) + img.astype(np.float32) * orig_weight
    output = output.astype(np.uint8).copy()

    output = cv2.erode(output, kernel_smoothing, iterations=erode)
    output = cv2.dilate(output, kernel_smoothing, iterations=dilate * 2)
    output = cv2.erode(output, kernel_smoothing, iterations=erode)

    return output


def isiterable(x):
    return hasattr(x, "__iter__")


def run(
    src: Image.Image,
    target_size: int = 256,
    patch_size: int = 8,
    thickness: int = 2,
    colors: int = 0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    mode: str = "contrast",
    no_color_matching: bool = False,
    no_upscale: bool = False,
    no_downscale: bool = False,
) -> Tuple[Image.Image, List[List[Union[str, float]]]]:
    img = np.array(src)
    H, W, _ = img.shape

    ratio = W / H
    if isiterable(target_size) and len(target_size) > 1:
        target_org_hw = tuple([int(i * patch_size) for i in target_size][:2])
        ratio = target_org_hw[0] / target_org_hw[1]
        target_org_size = target_org_hw[1]
        target_size = ((target_org_size**2) / (patch_size**2) * ratio) ** 0.5
    else:
        if isiterable(target_size):
            target_size = target_size[0]
        target_org_size = (target_size**2 * patch_size**2 / ratio) ** 0.5
        target_org_hw = (int(target_org_size * ratio), int(target_org_size))

    img = cv2.resize(img, target_org_hw)
    org_img = img.copy()

    if thickness:
        img = outline_expansion(img, thickness, thickness, patch_size, 9, 4)

    if no_color_matching is not True:
        img = match_color(img, org_img)

    if no_downscale:
        return img
    img_sm = downscale_mode[mode](img, target_size)

    if colors:
        img_sm = kmeans_color_quant(img_sm, colors)

    if contrast != 1 or saturation != 1:
        img_sm = color_styling(img_sm, saturation, contrast)

    if no_upscale:
        return img_sm

    img_lg = cv2.resize(img_sm, (W, H), interpolation=cv2.INTER_NEAREST)
    return img_lg
