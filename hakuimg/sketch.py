from PIL import Image, ImageFilter, ImageEnhance, ImageColor
import cv2
import numpy as np



def fix_float(val, eps=1e-3):
    return float(val)-eps

def dog_filter(img, kernel=0, sigma=1.4, k_sigma=1.6, gamma=1):
    g1 = cv2.GaussianBlur(img, (kernel,kernel), sigma)
    g2 = cv2.GaussianBlur(img, (kernel,kernel), sigma*k_sigma)
    return g1 - fix_float(gamma) * g2

def xdog(img, sigma, k_sigma, eps, phi, gamma, color, scale=True):
    if color=='gray':
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    dog = dog_filter(img, 0, sigma, k_sigma, gamma)
    dog = dog/dog.max()
    e = 1+np.tanh(fix_float(phi) * (dog-fix_float(eps)))
    e[e>=1] = 1
    
    if color=='gray':
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    if scale:
        return Image.fromarray((e*255).astype('uint8'))
    else:
        return Image.fromarray(e.astype('uint8')*255)


def run(*args):
    return xdog(*args)