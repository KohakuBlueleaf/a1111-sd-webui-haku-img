import torch
import numpy as np
from PIL import Image


def tensor_to_pil(image: torch.Tensor):
    return Image.fromarray(
        np.clip(255.0 * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
    )


def pil_to_tensor(image: Image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)


def image_preprocess(img: torch.Tensor, device: str):
    use_channel_last = False
    if img.ndim == 3:
        img = img.unsqueeze(0)
    if img.size(3) <= 4:
        img = img.permute(0, 3, 1, 2)
        use_channel_last = True
    if img.size(1) == 4:
        img = img[:, :3]
    org_device = img.device
    if device != "default":
        img = img.to(device)
    return img, use_channel_last, org_device
