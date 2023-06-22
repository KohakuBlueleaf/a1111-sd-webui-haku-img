import enum

import numpy as np


class Axis(str, enum.Enum):
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"


def run(np_img, axis):
    if axis == Axis.VERTICAL:
        np_img = np.flipud(np_img)
    elif axis == Axis.HORIZONTAL:
        np_img = np.fliplr(np_img)

    return np_img
