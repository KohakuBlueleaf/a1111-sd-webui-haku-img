from PIL import ImageDraw, Image, ImageFilter


def get_vignette(img, value):
    if value <= 0:
        return img

    width, height = img.size
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    padding = 100 - value * 100
    draw.ellipse(
        (-padding, -padding, width + padding, height + padding), fill=255
    )
    mask = mask.filter(ImageFilter.GaussianBlur(radius=100))
    return Image.composite(img, Image.new("RGB", img.size, "black"), mask)
