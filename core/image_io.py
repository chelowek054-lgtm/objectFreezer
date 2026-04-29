import base64
import io
from typing import Any

import torch

from .deps import require_pil_image


def tensor_to_base64_png(image_chw_or_hwc: torch.Tensor) -> str:
    Image = require_pil_image()

    t = image_chw_or_hwc.detach().to("cpu")
    if t.ndim == 3 and t.shape[-1] in (3, 4):
        hwc = t
    elif t.ndim == 3 and t.shape[0] in (3, 4):
        hwc = t.permute(1, 2, 0)
    else:
        raise ValueError("Unsupported image tensor shape: {}".format(tuple(t.shape)))

    if hwc.shape[-1] == 4:
        hwc = hwc[:, :, :3]

    hwc = hwc.clamp(0, 1)
    arr = (hwc * 255.0).round().to(torch.uint8).numpy()
    im = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def vae_encode_all_frames(images: torch.Tensor, vae: Any) -> torch.Tensor:
    """
    Encode every IMAGE row to latent [N, C, H_lat, W_lat] float32 (same as VAE.encode batch).

    images: ComfyUI IMAGE [N, H, W, C] float [0..1].
    """
    if images.ndim != 4:
        raise ValueError("Expected IMAGE tensor [N,H,W,C], got {}".format(tuple(images.shape)))
    pixels = images[:, :, :, :3]
    latent = vae.encode(pixels)
    if latent.ndim != 4:
        raise ValueError("Unexpected VAE latent shape: {}".format(tuple(latent.shape)))
    return latent.detach().float().contiguous()


def vae_encode_pool(images: torch.Tensor, vae: Any) -> torch.Tensor:
    """
    images: ComfyUI IMAGE => [B,H,W,C] float [0..1]
    returns: z_vision => [C_latent] float32 on CPU
    """
    if images.ndim != 4:
        raise ValueError("Expected IMAGE tensor [B,H,W,C], got {}".format(tuple(images.shape)))
    pixels = images[:, :, :, :3]
    latent = vae.encode(pixels)  # [B,C,h,w]
    if latent.ndim != 4:
        raise ValueError("Unexpected VAE latent shape: {}".format(tuple(latent.shape)))
    z = latent.mean(dim=(2, 3))  # [B,C]
    z = z.mean(dim=0)  # [C]
    return z.detach().to(dtype=torch.float32, device="cpu").contiguous()

