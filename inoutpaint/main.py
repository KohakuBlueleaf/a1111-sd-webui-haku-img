from __future__ import annotations

import numpy as np
import cv2
from PIL import Image

try:
    from .utils import *
except:
    from utils import *


HTML_TEMPLATES = {"resolution": """<textarea>"""}


def run(img, w, h, t, b, l, r, mode="fill"):
    w, h, t, b, l, r = int(w), int(h), int(t), int(b), int(l), int(r)
    
    img = img.resize((r-l, b-t))
    new_img, mask = resize_with_mask(
        img, w, h, t, b, l, r
    )

    w, h = img.size
    w_n, h_n = new_img.size

    if t:
        new_img.paste(img.resize((w, t), box=(0, 0, w, 0)), box=(l, 0))
    if b<h_n:
        new_img.paste(img.resize((w, h_n-b), box=(0, h, w, h)), box=(l, b))

    if l:
        new_img.paste(new_img.resize((l, h_n), box=(l + 1, 0, l + 1, h_n)), box=(0, 0))
    if r<w_n:
        new_img.paste(
            new_img.resize((w_n-r, h_n), box=(w + l - 1, 0, w + l - 1, h_n)), box=(l + w, 0)
        )

    return new_img, mask, f"{w_n} x {h_n}"


if __name__ == "__main__":
    img = Image.open(
        "./01491-2023-01-21_d7ff2a1d60_KBlueLeaf_KBlueLeaf_856221927-768x512.png"
    )
    new, mask, _ = run(np.array(img), 64, 192, 64, 64)

    w, h = new.size

    if w > h:
        demo = Image.new("RGB", (w, h * 2))
        demo.paste(new, box=(0, 0))
        demo.paste(mask, box=(0, h))
    else:
        demo = Image.new("RGB", (w * 2, h))
        demo.paste(new, box=(0, 0))
        demo.paste(mask, box=(w, 0))
    new.save("./test-img.png")
    mask.save("./test-mask.png")
    demo.show()
