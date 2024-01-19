from PIL import Image, ImageFilter


def run(img, img_blur):
    blur = ImageFilter.GaussianBlur(img_blur)
    return img.filter(blur)
