import re
from typing import Any, Dict, List

import torch


def extract_geo(details: Dict[str, Any], summary_text: str, d_geo: int) -> torch.Tensor:
    vals = []  # type: List[float]

    preferred_keys = [
        "height",
        "width",
        "length",
        "depth",
        "shoulder_width",
        "head_to_body_ratio",
        "waist",
        "hips",
    ]
    for k in preferred_keys:
        v = details.get(k, None)
        if isinstance(v, (int, float)):
            vals.append(float(v))

    if len(vals) < d_geo and isinstance(summary_text, str) and summary_text:
        for m in re.finditer(r"(-?\d+(?:\.\d+)?)\s*(cm|mm|m|%)?", summary_text.lower()):
            num = float(m.group(1))
            unit = (m.group(2) or "").strip()
            if unit == "mm":
                num = num / 1000.0
            elif unit == "cm":
                num = num / 100.0
            elif unit == "%":
                num = num / 100.0
            vals.append(num)
            if len(vals) >= d_geo:
                break

    out = torch.zeros((d_geo,), dtype=torch.float32, device="cpu")
    if vals:
        out[: min(d_geo, len(vals))] = torch.tensor(vals[:d_geo], dtype=torch.float32, device="cpu")
    return out.contiguous()

