import enum

import cv2
import numpy as np
from PIL import Image


def run(np_img, k1, k2):
    height, width = np_img.shape[:2]
    focal_length = width
    center_x = width / 2
    center_y = height / 2

    K = np.array(
        [[focal_length, 0, center_x], [0, focal_length, center_y], [0, 0, 1]], dtype=np.float64,
    )
    D = np.array([k1, k2, 0, 0], dtype=np.float64)
    img = cv2.fisheye.undistortImage(np_img, K, D, Knew=K)

    return Image.fromarray(img)
