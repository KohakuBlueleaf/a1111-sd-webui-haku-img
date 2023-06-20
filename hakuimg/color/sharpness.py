from PIL import ImageEnhance


def get_sharpness(img, value):
    if value <= 0:
        return img

    return ImageEnhance.Sharpness(img).enhance((value+1) * 1.5)
