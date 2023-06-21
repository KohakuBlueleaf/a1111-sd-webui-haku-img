"""
Original: https://github.com/andrewdcampbell/tilt-shift/
"""
import cv2
import numpy as np


def tilt_shift(im, focus_height: int, dof: int = 60):
    if focus_height < 2*dof:
        focus_height = 2*dof
    if focus_height > im.shape[0] - 2*dof:
        focus_height = im.shape[0] - 2*dof

    above_focus, below_focus = im[:focus_height,:], im[focus_height:,:]
    above_focus = increasing_blur(above_focus[::-1,...], dof)[::-1,...]
    below_focus = increasing_blur(below_focus, dof)
    out = np.vstack((above_focus, below_focus))

    return out

def increasing_blur(im, dof=60):
    BLEND_WIDTH = dof
    blur_region = cv2.GaussianBlur(im[dof:,:], ksize=(15,15), sigmaX=0)
    if blur_region.shape[0] > dof*2:
        blur_region = increasing_blur(blur_region, dof)
    blend_col = np.linspace(1.0, 0, num=BLEND_WIDTH)
    blend_mask = np.tile(blend_col, (im.shape[1], 1)).T
    res = np.zeros_like(im)
    res[:dof,:] = im[:dof,:]
    # alpha blend region of width BLEND_WIDTH to hide seams between blur layers
    res[dof:dof+BLEND_WIDTH,:] = im[dof:dof+BLEND_WIDTH,:] * blend_mask[:, :, None] + \
        blur_region[:BLEND_WIDTH,:] * (1-blend_mask[:, :, None])
    res[dof+BLEND_WIDTH:,:] = blur_region[BLEND_WIDTH:]
    return res
