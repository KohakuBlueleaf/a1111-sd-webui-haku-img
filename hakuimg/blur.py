from PIL import Image, ImageFilter


def run(img3, img_blur):
    img = Image.fromarray(img3)
    blur = ImageFilter.GaussianBlur(img_blur)
    return img.filter(blur)