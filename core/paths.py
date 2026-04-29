import os
import re
from typing import Optional

import folder_paths


def plugin_root() -> str:
    return os.path.dirname(os.path.abspath(__file__ + "/.."))


def matrices_dir() -> str:
    d = os.path.join(plugin_root(), "matrices")
    os.makedirs(d, exist_ok=True)
    return d


def blueprints_output_dir(output_dir: Optional[str]) -> str:
    """
    Default is ComfyUI/output/blueprints.
    - If output_dir is relative (e.g. 'output/blueprints'), it is resolved from the ComfyUI root.
    """
    comfy_root = os.path.dirname(folder_paths.get_input_directory())
    if output_dir and output_dir.strip():
        d = output_dir.strip()
        if not os.path.isabs(d):
            d = os.path.join(comfy_root, d)
    else:
        d = os.path.join(folder_paths.get_output_directory(), "blueprints")
    os.makedirs(d, exist_ok=True)
    return d


def sanitize_filename(s: str) -> str:
    s = s.strip() or "object"
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s[:128]

