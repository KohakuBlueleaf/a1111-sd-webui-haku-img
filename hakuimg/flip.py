import enum

import numpy as np
from PIL import Image


class Axis(str, enum.Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


def run(pil_img, axis):
    np_img = np.array(pil_img)
    if axis == Axis.VERTICAL:
        np_img = np.flipud(np_img)
    elif axis == Axis.HORIZONTAL:
        np_img = np.fliplr(np_img)

    return Image.fromarray(np_img)
