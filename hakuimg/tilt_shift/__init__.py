import cv2
from PIL import Image

from .utils import tilt_shift


def run(np_img, focus_ratio: float, dof: int):
    focus_ratio += 5

    height = np_img.shape[0]

    focus_height = round(height * (focus_ratio / 10))
    np_img = tilt_shift(np_img, dof=dof, focus_height=focus_height)

    return Image.fromarray(np_img)
