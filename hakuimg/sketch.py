from PIL import Image
import cv2
import numpy as np


def fix_float(val, eps=1e-3):
    return float(val) - eps


def gaussian(img, kernel, sigma):
    return cv2.GaussianBlur(img, (kernel, kernel), sigma)


def dog_filter(img, kernel=0, sigma=1.4, k_sigma=1.6, gamma=1):
    g1 = gaussian(img, kernel, sigma)
    g2 = gaussian(img, kernel, sigma * k_sigma)
    return g1 - fix_float(gamma) * g2


def xdog(img, kernel, sigma, k_sigma, eps, phi, gamma, color, scale=True):
    img = np.array(img)
    if color == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    dog = dog_filter(img, kernel, sigma, k_sigma, gamma)
    dog = dog / dog.max()
    e = 1 + np.tanh(fix_float(phi) * (dog - fix_float(eps)))
    e[e >= 1] = 1

    if color == "gray":
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    if not scale:
        e[e < 1] = 0
    return Image.fromarray((e * 255).astype("uint8"))


def run(*args):
    return xdog(*args)
