from PIL import Image

from .kromo import add_chromatic


def run(pil_img, strength, blur=False):
    if strength <= 0:
        return pil_img

    img = pil_img

    if img.size[0] % 2 == 0 or img.size[1] % 2 == 0:
        if img.size[0] % 2 == 0:
            img = img.crop((0, 0, img.size[0] - 1, img.size[1]))
            img.load()
        if img.size[1] % 2 == 0:
            img = img.crop((0, 0, img.size[0], img.size[1] - 1))
            img.load()

    img = add_chromatic(img, strength + 0.12, not blur)
    return img
