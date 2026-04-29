import json
import os
from typing import Any, Dict, Optional

import torch

from .deps import require_safetensors


def load_zhyb_from_blueprint_path(blueprint_path: str) -> torch.Tensor:
    p = str(blueprint_path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.lower().endswith(".json"):
        raise RuntimeError("JSON blueprint format is not supported. Use .blueprint (safetensors).")

    safetensors = require_safetensors()
    t = safetensors.torch.load_file(p, device="cpu")

    if "z_hyb" not in t:
        raise RuntimeError("safetensors blueprint missing required tensor 'z_hyb'")

    z_hyb = t["z_hyb"]
    return z_hyb.detach().to("cpu", dtype=torch.float32).reshape(-1).contiguous()


def resolve_blueprint_from_index_json(index_json_path: str) -> str:
    """
    Returns an absolute path to the latest referenced .blueprint in an index JSON.
    Expected structure:
      {"entries": [{"file": "<name>.blueprint", ...}, ...]}
    """
    p = str(index_json_path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", None) if isinstance(data, dict) else None
    if not isinstance(entries, list) or len(entries) == 0:
        raise RuntimeError("index.json has no entries")
    last = entries[-1]
    fname = last.get("file", None) if isinstance(last, dict) else None
    if not isinstance(fname, str) or fname.strip() == "":
        raise RuntimeError("index.json last entry missing 'file'")
    return os.path.join(os.path.dirname(p), fname)


def load_blueprint_text_tokens(blueprint_path: str) -> Optional[torch.Tensor]:
    """Full text encoder sequence [1, N, D_te], float32 CPU. Missing → None."""
    p = str(blueprint_path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.lower().endswith(".json"):
        raise RuntimeError("Expected .blueprint (safetensors), got JSON path.")

    safetensors = require_safetensors()
    t = safetensors.torch.load_file(p, device="cpu")

    if "blueprint_text_tokens" not in t:
        return None

    seq = t["blueprint_text_tokens"]
    if not isinstance(seq, torch.Tensor):
        return None
    seq = seq.detach().to(dtype=torch.float32, device="cpu").contiguous()
    if seq.ndim == 2:
        seq = seq.unsqueeze(0)
    if seq.ndim != 3:
        raise RuntimeError("blueprint_text_tokens must be [1, N, D] or [N, D], got shape {}".format(tuple(seq.shape)))
    return seq


def load_reference_latent(blueprint_path: str) -> torch.Tensor:
    """
    Required reference latent ``reference_latent``: [1, C, H, W], float16/float32 CPU.
    Missing → RuntimeError.
    """
    p = str(blueprint_path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.lower().endswith(".json"):
        raise RuntimeError("Expected .blueprint (safetensors), got JSON path.")

    safetensors = require_safetensors()
    t = safetensors.torch.load_file(p, device="cpu")

    if "reference_latent" not in t:
        raise RuntimeError(
            "safetensors blueprint missing required tensor 'reference_latent' (re-export with Blueprint Creator)"
        )

    lat = t["reference_latent"]
    if not isinstance(lat, torch.Tensor):
        raise RuntimeError("reference_latent is not a tensor")
    lat = lat.detach().to(device="cpu").contiguous()
    if lat.ndim == 3:
        lat = lat.unsqueeze(0)
    if lat.ndim != 4:
        raise RuntimeError(
            "reference_latent must be [1,C,H,W] or [C,H,W], got {}".format(tuple(lat.shape))
        )
    if int(lat.shape[0]) != 1:
        raise RuntimeError("reference_latent must have batch==1, got shape {}".format(tuple(lat.shape)))
    return lat


def load_keyword_embedding(blueprint_path: str) -> Optional[torch.Tensor]:
    """Returns 1D float32 CPU tensor if ``keyword_embedding`` is present; else None."""
    p = str(blueprint_path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.lower().endswith(".json"):
        raise RuntimeError("Expected .blueprint (safetensors), got JSON path.")

    safetensors = require_safetensors()
    t = safetensors.torch.load_file(p, device="cpu")
    kw = t.get("keyword_embedding", None)
    if not isinstance(kw, torch.Tensor):
        return None
    return kw.detach().to("cpu", dtype=torch.float32).reshape(-1).contiguous()


def load_blueprint_metadata(blueprint_path: str) -> Dict[str, str]:
    """Loads safetensors metadata dict."""
    p = str(blueprint_path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    if p.lower().endswith(".json"):
        return {}

    try:
        from safetensors import safe_open
    except Exception:
        return {}

    with safe_open(p, framework="pt", device="cpu") as f:
        meta = f.metadata()
    return meta or {}


def save_blueprint_safetensors(path: str, tensors: Dict[str, torch.Tensor], metadata: Dict[str, str]) -> None:
    safetensors = require_safetensors()
    safetensors.torch.save_file(tensors, path, metadata=metadata)


def update_index_json(index_path: str, blueprint_filename: str, vlm_meta: Dict[str, Any]) -> None:
    entry = {"file": blueprint_filename, "vlm": vlm_meta}

    if os.path.exists(index_path):
        try:
            with open(index_path, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            existing = None
    else:
        existing = None

    if isinstance(existing, dict) and isinstance(existing.get("entries"), list):
        payload = existing
        payload["entries"].append(entry)
    else:
        payload = {"schema_version": 1, "entries": [entry]}

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
