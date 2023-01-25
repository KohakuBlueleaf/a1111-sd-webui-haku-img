from __future__ import annotations

from typing import Any
from numpy.typing import NDArray

import cv2
from PIL import Image
import numpy as np
import scipy as sp

from hakuimg.blend import Blend


def run(img, blur, strength, mode='BS'):
    img = img/255
    
    if mode == 'BS':
        img_blur = cv2.GaussianBlur(img, (0, 0), blur)
        img_glow = np.clip(Blend.screen(img_blur, img, strength), 0, 1)
    elif mode == 'BMBL':
        img_blur = cv2.GaussianBlur(img, (0, 0), blur)
        img_mul = np.clip(Blend.multiply(img_blur, img), 0, 1)
        img_mul_blur = cv2.GaussianBlur(img_mul, (0, 0), blur)
        img_glow = np.clip(Blend.lighten(img_mul_blur, img, strength), 0, 1)
    else:
        raise NotImplementedError
    
    return (img_glow*255).astype(np.uint8)