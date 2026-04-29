from typing import Optional

try:
    import safetensors.torch
except Exception as e:  # pragma: no cover
    safetensors = None
    safetensors_import_error = e

try:
    from PIL import Image
except Exception as e:  # pragma: no cover
    Image = None
    pil_import_error = e


def require_safetensors():
    if safetensors is None:  # pragma: no cover
        raise RuntimeError("safetensors is required but failed to import: {}".format(safetensors_import_error))
    return safetensors


def require_pil_image():
    if Image is None:  # pragma: no cover
        raise RuntimeError("Pillow (PIL) is required but failed to import: {}".format(pil_import_error))
    return Image

