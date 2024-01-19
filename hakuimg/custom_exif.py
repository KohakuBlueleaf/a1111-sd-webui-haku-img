from PIL import Image


def run(img, text):
    if not text:
        return img

    img.info["parameters"] = text
    return img
