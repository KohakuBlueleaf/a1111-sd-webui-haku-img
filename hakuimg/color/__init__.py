from PIL import Image, ImageEnhance
import cv2
import numpy as np

from .exposure_offset import get_exposure_offset
from .hdr import get_hdr
from .noise import get_noise
from .sharpness import get_sharpness
from .vignette import get_vignette


def run(
    img1,
    bright,
    contrast,
    sat,
    temp,
    hue,
    gamma,
    exposure_offset,
    hdr,
    noise,
    sharpness,
    vignette,
):
    bright /=100
    contrast /=100
    temp /=100
    sat /=100

    res = Image.fromarray(img1)

    res = get_exposure_offset(res, exposure_offset, bright)
    res = get_hdr(res, hdr, img1)
    res = get_sharpness(res, sharpness)
    res = get_noise(res, noise)
    res = get_vignette(res, vignette)

    #brigtness
    #res = Image.fromarray(res)
    brightness = ImageEnhance.Brightness(res)
    res = brightness.enhance(1+bright)

    #contrast
    cont = ImageEnhance.Contrast(res)
    res = cont.enhance(1+contrast)
    res = np.array(res).astype(np.float32)

    #temp
    if temp>0:
        res[:, :, 0] *= 1+temp
        res[:, :, 1] *= 1+temp*0.4
    elif temp<0:
        res[:, :, 2] *= 1-temp
    res = np.clip(res, 0, 255)/255

    res = np.clip(np.power(res, gamma), 0, 1)

    #saturation
    print(res.shape)
    sat_real = 1 + sat
    hls_img = cv2.cvtColor(res, cv2.COLOR_RGB2HLS)
    hls_img[:, :, 2] = np.clip(sat_real*hls_img[:, :, 2], 0, 1)
    res = cv2.cvtColor(hls_img, cv2.COLOR_HLS2RGB)*255

    # hue
    hsv_img = cv2.cvtColor(res, cv2.COLOR_RGB2HSV)
    print(np.max(hsv_img[:, :, 0]), np.max(hsv_img[:, :, 1]), np.max(hsv_img[:, :, 2]))
    hsv_img[:, :, 0] = (hsv_img[:, :, 0]+hue)%360

    res = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2RGB)

    res = res.astype(np.uint8)
    res = Image.fromarray(res, mode='RGB')
    return res
