from PIL import Image
from pixeloe.torch.pixelize import pixelize
from .image_preprocess import image_preprocess, pil_to_tensor, tensor_to_pil


def run(
    img: Image,
    pixel_size: int,
    thickness: int,
    num_colors: int,
    mode: str,
    quant_mode: str,
    dither_mode: str,
    device: str,
    color_quant: bool,
    no_post_upscale: bool,
):
    img = pil_to_tensor(img)
    img, use_channel_last, org_device = image_preprocess(img, device)
    result, _, _ = pixelize(
        img,
        pixel_size,
        thickness,
        mode,
        do_color_match=True,
        do_quant=color_quant,
        num_colors=num_colors,
        quant_mode=quant_mode,
        dither_mode=dither_mode,
        no_post_upscale=no_post_upscale,
        return_intermediate=True,
    )
    result = result.to(org_device)
    if use_channel_last:
        result = result.permute(0, 2, 3, 1)
    result = tensor_to_pil(result)
    return result
