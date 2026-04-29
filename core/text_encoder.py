from typing import Any

import torch


def encode_blueprint_text_tokens_for_diffusion(
    text_encoder: Any,
    text: str,
) -> torch.Tensor:
    """
    Raw TE conditioning from encode_from_tokens(..., return_pooled=False).

    Returns float32 CPU [1, T, D] with D equal to the text encoder output width (e.g. 12288 for Klein).
    No Linear projection and no Klein block slicing.
    """
    if text_encoder is None:
        raise RuntimeError("text_encoder is None")
    if not hasattr(text_encoder, "tokenize") or not hasattr(text_encoder, "encode_from_tokens"):
        raise RuntimeError("text_encoder must expose tokenize() and encode_from_tokens()")
    if not hasattr(text_encoder, "load_model"):
        raise RuntimeError("text_encoder must expose load_model() (ComfyUI CLIP)")

    tokens = text_encoder.tokenize(text)
    text_encoder.load_model(tokens)
    text_encoder.cond_stage_model.reset_clip_options()
    if getattr(text_encoder, "layer_idx", None) is not None:
        text_encoder.cond_stage_model.set_clip_options({"layer": text_encoder.layer_idx})
    text_encoder.cond_stage_model.set_clip_options({"execution_device": text_encoder.patcher.load_device})

    cond = text_encoder.encode_from_tokens(tokens, return_pooled=False)
    if not isinstance(cond, torch.Tensor):
        raise RuntimeError("encode_from_tokens did not return cond tensor")

    cond = cond.detach().float()
    return cond.cpu().contiguous()


def encode_text_sequence(text_encoder: Any, text: str) -> torch.Tensor:
    """
    Returns a sequence embedding (cond) as float32 CPU.
    Expected shape: [1, T, D] or [T, D].
    """
    if text_encoder is None:
        raise RuntimeError("text_encoder is None")

    if hasattr(text_encoder, "tokenize") and hasattr(text_encoder, "encode_from_tokens"):
        tokens = text_encoder.tokenize(text)
        try:
            cond, _pooled = text_encoder.encode_from_tokens(tokens, return_pooled="unprojected")
        except TypeError:
            cond, _pooled = text_encoder.encode_from_tokens(tokens, return_pooled=True)

        if not isinstance(cond, torch.Tensor):
            raise RuntimeError("encode_from_tokens did not return cond tensor")

        return cond.detach().to(dtype=torch.float32, device="cpu").contiguous()

    raise RuntimeError("Unsupported text_encoder API: expected tokenize() and encode_from_tokens()")


def encode_text_embedding(text_encoder: Any, text: str) -> torch.Tensor:
    """
    Returns 1D float32 embedding on CPU.
    Works with ComfyUI CLIP-like objects that expose tokenize() and encode_from_tokens().
    """
    if text_encoder is None:
        raise RuntimeError("text_encoder is None")

    if hasattr(text_encoder, "tokenize") and hasattr(text_encoder, "encode_from_tokens"):
        tokens = text_encoder.tokenize(text)
        try:
            cond, pooled = text_encoder.encode_from_tokens(tokens, return_pooled="unprojected")
        except TypeError:
            cond, pooled = text_encoder.encode_from_tokens(tokens, return_pooled=True)

        if isinstance(pooled, torch.Tensor):
            emb = pooled
        elif isinstance(cond, torch.Tensor):
            if cond.ndim == 3:
                emb = cond.mean(dim=1).squeeze(0)
            elif cond.ndim == 2:
                emb = cond.mean(dim=0)
            else:
                emb = cond.reshape(-1)
        else:
            raise RuntimeError("encode_from_tokens did not return tensors")

        return emb.detach().to(dtype=torch.float32, device="cpu").contiguous()

    raise RuntimeError("Unsupported text_encoder API: expected tokenize() and encode_from_tokens()")
