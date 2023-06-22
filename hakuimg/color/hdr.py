import cv2
import numpy as np
from PIL import ImageFilter, ImageChops, Image, ImageOps, ImageEnhance

from blendmodes.blend import blendLayers, BlendType


def get_hdr(img, value, original_img):
    if value <= 0:
        return img

    blurred = img.filter(ImageFilter.GaussianBlur(radius=2.8))
    difference = ImageChops.difference(img, blurred)
    sharp_edges = Image.blend(img, difference, 1)

    converted_original_img = (
        np.array(original_img)[:, :, ::-1].copy().astype("float32") / 255.0
    )
    converted_sharped = (
        np.array(sharp_edges)[:, :, ::-1].copy().astype("float32") / 255.0
    )

    color_dodge = converted_original_img / (1 - converted_sharped)
    converted_color_dodge = (255 * color_dodge).clip(0, 255).astype(np.uint8)

    temp_img = Image.fromarray(
        cv2.cvtColor(converted_color_dodge, cv2.COLOR_BGR2RGB)
    )
    inverted_color_dodge = ImageOps.invert(temp_img)
    black_white_color_dodge = ImageEnhance.Color(inverted_color_dodge).enhance(0)
    hue = blendLayers(temp_img, black_white_color_dodge, BlendType.HUE)
    hdr_image = blendLayers(hue, temp_img, BlendType.NORMAL, 0.7)

    return blendLayers(img, hdr_image, BlendType.NORMAL, value * 2).convert(
        "RGB"
    )
