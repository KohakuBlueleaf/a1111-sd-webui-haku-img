from PIL import Image
from .image_preprocess import image_preprocess, pil_to_tensor, tensor_to_pil
from pixeloe.torch.outline import outline_expansion


def run(
    img: Image,
    pixel_size: int,
    thickness: int,
    device: str,
):
    img = pil_to_tensor(img)
    img, use_channel_last, org_device = image_preprocess(img, device)
    oe_image, _ = outline_expansion(img, thickness, thickness, pixel_size)
    oe_image = oe_image.to(org_device)
    if use_channel_last:
        oe_image = oe_image.permute(0, 2, 3, 1)
    oe_image = tensor_to_pil(oe_image)
    return oe_image
