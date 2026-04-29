import datetime as _dt
import json
import logging
import os
from typing import Any, Dict, Optional

import torch

from ..core.blueprint_io import save_blueprint_safetensors, update_index_json
from ..core.geo import extract_geo
from ..core.image_io import tensor_to_base64_png, vae_encode_first_frame, vae_encode_pool
from ..core.matrices import load_or_create_R
from ..core.paths import blueprints_output_dir, sanitize_filename
from ..core.text_encoder import (
    encode_blueprint_text_tokens_for_diffusion,
    encode_text_embedding,
    encode_text_sequence,
)
from ..core.vlm_ollama import ollama_chat


def _project_semantic(emb: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    if emb.ndim != 1:
        emb = emb.reshape(-1)
    if R.shape[1] != emb.shape[0]:
        raise ValueError("Projection shape mismatch: R={} emb={}".format(tuple(R.shape), tuple(emb.shape)))
    return (R @ emb).to(dtype=torch.float32, device="cpu").contiguous()


_D_SEM = 1024
_D_GEO = 20
_D_FACE = 512

logger = logging.getLogger(__name__)


class BlueprintCreator:
    CATEGORY = "blueprint"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "vae": ("VAE",),
                "text_encoder": ("CLIP",),
                "ollama_url": ("STRING", {"default": "http://127.0.0.1:11434"}),
                "vlm_model": ("STRING", {"default": "qwen3-vl"}),
                "object_id": ("STRING", {"default": "object"}),
                "object_class": (["character", "building", "prop"], {"default": "prop"}),
            },
            "optional": {
                "output_dir": ("STRING", {"default": ""}),
                "seed": ("INT", {"default": 12345, "min": 0, "max": 2**31 - 1}),
                "num_text_tokens": ("INT", {"default": 16, "min": 1, "max": 256, "step": 1}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("blueprint_path",)
    FUNCTION = "create"

    def create(
        self,
        images,
        vae,
        text_encoder,
        ollama_url,
        vlm_model,
        object_id,
        object_class,
        output_dir="",
        seed=12345,
        num_text_tokens=16,
    ):
        lat_ref = vae_encode_first_frame(images, vae)
        reference_latent_f16 = lat_ref.detach().half().cpu().contiguous()
        reference_latent_meta = {
            "tensor_key": "reference_latent",
            "layout": "[batch, channels, height, width]",
            "batch": 1,
            "channels": int(reference_latent_f16.shape[1]),
            "height_lat": int(reference_latent_f16.shape[2]),
            "width_lat": int(reference_latent_f16.shape[3]),
            "note": "Reference is mandatory; pick blueprint entry (index.json) in Blueprint Injector.",
        }

        z_vision = vae_encode_pool(images, vae)

        b = int(images.shape[0])
        images_b64 = [tensor_to_base64_png(images[i]) for i in range(b)]
        vlm = ollama_chat(images_b64, ollama_url=ollama_url, model=vlm_model, object_class=object_class)
        summary_text = vlm.get("summary_text", "")
        details = vlm.get("details", {})  # type: Dict[str, Any]
        face_desc = vlm.get("face_description", None)  # type: Optional[str]

        # Use object_id as keyword/alias for text association.
        keyword = str(object_id)
        summary_text_prefixed = "{}: {}".format(keyword, summary_text)
        emb_sem = encode_text_embedding(text_encoder, summary_text_prefixed)
        R_sem = load_or_create_R("R_sem", out_dim=int(_D_SEM), in_dim=int(emb_sem.shape[0]), seed=int(seed))
        z_sem = _project_semantic(emb_sem, R_sem)

        text_seq = encode_blueprint_text_tokens_for_diffusion(text_encoder, summary_text_prefixed)
        if text_seq.ndim != 3:
            raise RuntimeError("blueprint text seq expected [1,T,D], got {}".format(tuple(text_seq.shape)))
        blueprint_text_tokens = text_seq.detach().to(dtype=torch.float32, device="cpu").contiguous()
        n_cap = int(num_text_tokens)
        if n_cap < 1:
            n_cap = 1
        blueprint_text_tokens = blueprint_text_tokens[:, :n_cap, :].contiguous()
        _t = int(blueprint_text_tokens.shape[1])
        _d = int(blueprint_text_tokens.shape[-1])
        logger.info(
            "BlueprintCreator blueprint_text_tokens shape=(%s, %s, %s) (raw TE cond)",
            int(blueprint_text_tokens.shape[0]),
            _t,
            _d,
        )
        if _d != 12288:
            logger.warning(
                "BlueprintCreator: full TE cond last dim is %s, not 12288 — "
                "use Qwen/Klein TE that outputs 3×4096 if you need Klein-compatible blueprints.",
                _d,
            )

        z_geo = extract_geo(details, summary_text, d_geo=int(_D_GEO))

        if object_class == "character" and face_desc:
            emb_face = encode_text_embedding(text_encoder, face_desc)
            R_face = load_or_create_R("R_face", out_dim=int(_D_FACE), in_dim=int(emb_face.shape[0]), seed=int(seed) + 1)
            z_face = _project_semantic(emb_face, R_face)
        else:
            z_face = torch.zeros((int(_D_FACE),), dtype=torch.float32, device="cpu")

        z_hyb = torch.cat([z_vision, z_sem, z_geo, z_face], dim=0).contiguous()

        # keyword embedding (sequence pooled) for cross-attn association
        kw_seq = encode_text_sequence(text_encoder, keyword)
        if kw_seq.ndim == 3:
            kw = kw_seq.mean(dim=1).squeeze(0)
        elif kw_seq.ndim == 2:
            kw = kw_seq.mean(dim=0)
        else:
            kw = kw_seq.reshape(-1)
        keyword_embedding = kw.detach().to(dtype=torch.float32, device="cpu").contiguous()

        now = _dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        out_dir = blueprints_output_dir(output_dir)
        # Safetensors only: separate files, per-character folder.
        fname = "{}_{}.blueprint".format(sanitize_filename(object_id), now)
        out_dir = os.path.join(out_dir, sanitize_filename(object_id))
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)

        dims_payload = {
            "d_vision": int(z_vision.shape[0]),
            "d_sem": int(_D_SEM),
            "d_geo": int(_D_GEO),
            "d_face": int(_D_FACE),
            "d_total": int(z_hyb.shape[0]),
            "blueprint_text_tokens_shape": list(blueprint_text_tokens.shape),
            "reference_latent": reference_latent_meta,
        }

        metadata = {
            "object_id": str(object_id),
            "object_class": str(object_class),
            "ollama_url": str(ollama_url),
            "vlm_model": str(vlm_model),
            "summary_text": str(summary_text),
            "summary_text_prefixed": str(summary_text_prefixed),
            "keyword": str(keyword),
            "keyword_pooling": "mean",
            "details_json": json.dumps(details, ensure_ascii=False),
            "face_description": face_desc or "",
            "created_at": now,
            "dims": json.dumps(dims_payload),
            "seed_R_sem": str(int(seed)),
            "seed_R_face": str(int(seed) + 1),
            "num_text_tokens": str(_t),
            "context_token_dim": "0",
            "has_reference_latent": "true",
        }

        tensors = {
            "z_vision": z_vision,
            "z_sem": z_sem,
            "z_geo": z_geo,
            "z_face": z_face,
            "z_hyb": z_hyb,
            "keyword_embedding": keyword_embedding,
            "blueprint_text_tokens": blueprint_text_tokens,
            "reference_latent": reference_latent_f16,
        }

        save_blueprint_safetensors(path, tensors=tensors, metadata=metadata)
        # Maintain index.json with VLM info and file references.
        index_path = os.path.join(out_dir, "{}.index.json".format(sanitize_filename(object_id)))
        vlm_meta = {
            "object_id": str(object_id),
            "object_class": str(object_class),
            "ollama_url": str(ollama_url),
            "vlm_model": str(vlm_model),
            "summary_text": str(summary_text),
            "summary_text_prefixed": str(summary_text_prefixed),
            "keyword": str(keyword),
            "has_keyword_embedding": True,
            "has_blueprint_text_tokens": True,
            "has_reference_latent": True,
            "num_text_tokens": _t,
            "context_token_dim": 0,
            "details": details,
            "face_description": face_desc or "",
            "created_at": now,
        }
        update_index_json(index_path, blueprint_filename=fname, vlm_meta=vlm_meta)

        return (path,)

