import cv2
from PIL import Image
import numpy as np
import scipy as sp

from hakuimg.blend import Blend


def run(pil_img, blur, strength, mode="BS"):
    img = np.array(pil_img)
    img = img / 255

    if mode == "BS":
        img_blur = cv2.GaussianBlur(img, (0, 0), blur)
        img_glow = Blend.screen(img_blur, img, strength)
    elif mode == "BMBL":
        img_blur = cv2.GaussianBlur(img, (0, 0), blur)
        img_mul = Blend.multiply(img_blur, img)
        img_mul_blur = cv2.GaussianBlur(img_mul, (0, 0), blur)
        img_glow = Blend.lighten(img_mul_blur, img, strength)
    else:
        raise NotImplementedError

    return (img_glow * 255).astype(np.uint8)
