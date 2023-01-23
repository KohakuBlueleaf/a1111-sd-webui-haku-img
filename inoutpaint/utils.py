from __future__ import annotations

import numpy as np
import cv2
from PIL import Image


def resize_with_mask(
    img: Image.Image, 
    expands: tuple[int, int, int, int]
) -> tuple[Image.Image, Image.Image]:
    w, h = img.size
    u, d, l, r = expands
    
    new_img = Image.new('RGB', (w+l+r, h+u+d))
    new_img.paste(img, (l, u))
    
    mask = Image.new('L', (w+l+r, h+u+d), 255)
    mask_none = Image.new('L', (w, h), 0)
    mask.paste(mask_none, (l, u))
    return new_img, mask