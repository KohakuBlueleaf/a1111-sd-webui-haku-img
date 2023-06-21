from PIL import Image, ImageFilter


def run(np_img, text):
    if not text:
        return np_img

    img = Image.fromarray(np_img)
    img.info["parameters"] = text

    return img
