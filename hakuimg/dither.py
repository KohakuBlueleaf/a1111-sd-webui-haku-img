from typing import Any
from numpy.typing import NDArray

from itertools import product

import cv2
from PIL import Image
import numpy as np


def dithering(
    img: NDArray[Any],
    find_new_color
):
    d_h, d_w, c = img.shape
    new_res = np.array(img, dtype=np.float32)/255
    for i, j in product(range(d_h), range(d_w)):
        old_val = new_res[i, j].copy()
        new_val = find_new_color(old_val)
        new_res[i, j] = new_val
        err = old_val - new_val
        
        if j < d_w - 1:
            new_res[i, j+1] += err * 7/16
        if i < d_h - 1:
            new_res[i+1, j] += err * 5/16
            if j > 0:
                new_res[i+1, j-1] += err * 3/16
            if j < d_w - 1:
                new_res[i+1, j+1] += err * 1/16
    return np.clip(new_res/np.max(new_res, axis=(0,1))*255, 0, 255)