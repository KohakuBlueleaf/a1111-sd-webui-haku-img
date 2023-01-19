from typing import Any
from numpy.typing import NDArray

import cv2
from PIL import Image
import numpy as np


INFLATE_FILTER = [
    None,
    np.array(
        [[0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]], np.uint8
    ),
    np.array(
        [[1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]], np.uint8
    ),
    np.array(
        [[0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0]], np.uint8
    ),
    np.array(
        [[1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1]], np.uint8
    ),
    np.ones((7, 7), np.uint8),
    np.ones((9, 9), np.uint8),
    np.ones((11, 11), np.uint8),
    np.ones((13, 13), np.uint8),
    np.ones((15, 15), np.uint8),
    np.ones((17, 17), np.uint8)
]


def read_img_as_array(
    img
) -> NDArray[Any]:
    '''Convert image to RGBA and read to ndarray'''
    img = Image.fromarray(img)
    img = img.convert('RGBA')
    img_arr = np.asarray(img)
    return img_arr


def preprocess(
    img: NDArray[Any],
    blur: int = 0, 
    erode: int = 0,
) -> NDArray[Any]:
    '''
    Process for 
    * outline inflation (erode)
    * smoothing (blur)
    * saturation
    * contrast
    '''
    # outline process
    if erode:
        img = cv2.erode(
            img, INFLATE_FILTER[erode], 
            iterations = 1,
        )
    
    # blur process
    if blur:
        img = cv2.bilateralFilter(
            img, 15, blur*20, 20
        )
    img = img.astype(np.float32)
    return img


def pixelize(
    img: NDArray[Any],
    k: int, c: int,
    d_w: int, d_h: int,
    o_w: int, o_h: int,
    precise: int,
) -> tuple[NDArray[Any], NDArray[Any]]:
    '''
    Use down scale and up scale to make pixel image.
    
    And use k-means to confine the num of colors.
    '''
    img = cv2.resize(
        img, (d_w, d_h), 
        interpolation = cv2.INTER_NEAREST
    )
    
    # reshape to 1-dim array(for every color) for k-means
    # use k-means to abstract the colors to use
    img_cp = img.reshape(-1, c)
    criteria = (
        cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 
        precise*5, 0.01
    )
    _, label, center = cv2.kmeans(
        img_cp, k, None,
        criteria, 1, cv2.KMEANS_PP_CENTERS
    )
    result = center[label.flatten()]
    result = result.reshape((img.shape))
    
    result = cv2.resize(
        result, 
        (o_w, o_h), 
        interpolation=cv2.INTER_NEAREST
    )
    return result.astype(np.uint8), center.astype(np.uint8)


def run(
    src: Image.Image,
    k: int = 3,
    scale: int = 2,
    blur: int = 0, 
    erode: int = 0,
    precise: int = 10,
) -> tuple[Image.Image, list[list[str|float]]]:
    #print('Start process.')
    #print('Read raw image... ', end='', flush=True)
    img = read_img_as_array(src)
    
    #convert color space
    alpha_channel = img[:, :, 3]
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    h, w, c = img.shape
    d_h = h // scale
    d_w = w // scale
    o_h = d_h * scale
    o_w = d_w * scale
    #print('done!')
    
    #print('Image preprocess... ', end='', flush=True)
    # preprocess(erode, blur, saturation, contrast)
    img = preprocess(
        img,
        blur, erode
    )
    #print('done!')
    
    #print('Pixelize... ', end='', flush=True)
    # pixelize(using k-means)
    result, colors = pixelize(
        img, k, c,
        d_w, d_h,
        o_w, o_h,
        precise
    )
    #print('done!')
    
    #print('Process output image... ', end='', flush=True)
    # add alpha channel
    a = cv2.resize(
        alpha_channel, (d_w, d_h), 
        interpolation = cv2.INTER_NEAREST
    )
    a = cv2.resize(
        a, (o_w, o_h), 
        interpolation = cv2.INTER_NEAREST
    )
    a[a!=0]=255
    if 0 not in a:
        a[0, 0] = 0
    r, g, b = cv2.split(result)
    result = cv2.merge((r, g, b, a))
    
    # for saving to png
    result = cv2.cvtColor(
        result, cv2.COLOR_RGBA2BGRA
    )
    #print('done!')
    
    return Image.fromarray(result)