import logging
import os
import json
from typing import List, Optional

import torch

import folder_paths

from ..core.blueprint_io import (
    load_blueprint_text_tokens,
    load_keyword_embedding,
    load_reference_latent,
    resolve_blueprint_from_index_json,
)
from ..core.matrices import load_or_create_W

import comfy.patcher_extension

logger = logging.getLogger(__name__)


def _tensor_debug_line(name: str, t: Optional[torch.Tensor]) -> str:
    if not isinstance(t, torch.Tensor):
        return f"{name}=None"
    td = t.detach().cpu().float()
    return (
        f"{name} shape={tuple(td.shape)} dtype={td.dtype} "
        f"min={float(td.min()):.6g} max={float(td.max()):.6g} mean={float(td.mean()):.6g}"
    )


def _list_blueprints_input_dir() -> List[str]:
    """
    Returns relative paths (from ComfyUI/output/blueprints) for dropdown.
    Includes: *.index.json
    """
    out_dir = folder_paths.get_output_directory()
    bp_dir = os.path.join(out_dir, "blueprints")
    if not os.path.isdir(bp_dir):
        return []
    files = []
    for root, _, filenames in os.walk(bp_dir):
        for f in filenames:
            p = os.path.join(root, f)
            if not os.path.isfile(p):
                continue
            lf = f.lower()
            if lf.endswith(".index.json"):
                rel = os.path.relpath(p, bp_dir)
                files.append(rel.replace("\\", "/"))
    files.sort()
    return files


def _resolve_blueprint_path(selection: str) -> str:
    s = str(selection).strip()
    if s == "":
        return ""

    # Absolute path: keep as-is.
    if os.path.isabs(s):
        return s

    # If user pasted a path that already exists (relative to cwd), allow it.
    if os.path.exists(s):
        return s

    # Otherwise treat it as a relative path under output/blueprints
    # (supports subfolders like 'hero/hero_....blueprint').
    return os.path.join(folder_paths.get_output_directory(), "blueprints", s)


def _resolve_blueprint_from_index_json_at(index_json_path: str, entry_index: int) -> str:
    """
    Resolve a .blueprint path from an index.json by entry index.
    If entry_index is out of range, it is clamped to [0, len(entries)-1].
    """
    p = str(index_json_path)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    entries = data.get("entries", None) if isinstance(data, dict) else None
    if not isinstance(entries, list) or len(entries) == 0:
        raise RuntimeError("index.json has no entries")
    i_raw = int(entry_index)
    i = max(0, min(i_raw, len(entries) - 1))
    e = entries[i]
    fname = e.get("file", None) if isinstance(e, dict) else None
    if not isinstance(fname, str) or fname.strip() == "":
        raise RuntimeError("index.json entry missing 'file' at index {}".format(i))
    return os.path.join(os.path.dirname(p), fname), i_raw, i, len(entries)


def _rebuild_txt_ids_for_new_length(
    txt_ids: torch.Tensor,
    L_new: int,
    atol: float = 1e-3,
) -> torch.Tensor:
    """
    Extend or shrink txt_ids along sequence dim to length L_new, following Flux._forward
    semantics when possible: axes that matched linspace(0, L_old-1) get full linspace(0, L_new-1).
    """
    if txt_ids.ndim != 3:
        raise ValueError("txt_ids must be [B, L, C]")
    B, L_old, C = txt_ids.shape
    device = txt_ids.device
    dtype = txt_ids.dtype
    if L_new == L_old:
        return txt_ids

    out = torch.zeros((B, L_new, C), device=device, dtype=dtype)
    if L_old <= 0:
        return out

    ref_col = txt_ids[0, :, :].to(dtype=torch.float32, device="cpu")

    for a in range(C):
        col_old = ref_col[:, a]
        if L_old > 1:
            expected = torch.linspace(0, L_old - 1, steps=L_old, dtype=torch.float32)
            is_linspace = torch.allclose(col_old, expected, rtol=0, atol=atol)
        else:
            is_linspace = False

        if is_linspace and L_new > 0:
            new_col = torch.linspace(0, L_new - 1, steps=L_new, device=device, dtype=dtype)
            out[:, :, a] = new_col
        else:
            take = min(L_old, L_new)
            if take > 0:
                out[:, :take, a] = txt_ids[:, :take, a]
    return out


def _blueprint_reference_latent_diffusion_wrapper(
    reference_latent_cpu: torch.Tensor,
    ref_latents_method_ui: str,
    ref_scale: float,
):
    """
    Flux ``diffusion_model`` wrapper: merge blueprint ``reference_latent`` into ``ref_latents``.

    Comfy passes ``ref_latents`` as the 6th positional arg to ``Flux._forward`` (after ``guidance``),
    not only in ``kwargs`` — we must merge into that slot to avoid duplicate keyword arguments.

    ``reference_latent_cpu`` is [1, C, H, W] on CPU (float16/float32). Maps UI method:
    ``default`` → Flux ``offset`` (spatial placement); ``index_timestep_zero`` → Kontext-style timestep split.
    """

    def _wrap(executor, *args, **kwargs):
        """
        Flux passes ref_latents positionally into the wrapper stack; extra conds may also provide it
        via kwargs. We must consolidate and call downstream without duplicate values.
        """
        if len(args) < 8:
            return executor(*args, **kwargs)

        x, timestep, context, y, guidance, ref_latents_pos, control, transformer_options = args[:8]

        kwargs = dict(kwargs)
        ref_latents_kw = kwargs.pop("ref_latents", None)

        existing_refs: list = []
        for existing in (ref_latents_pos, ref_latents_kw):
            if existing is None:
                continue
            if isinstance(existing, list):
                existing_refs.extend(existing)
            else:
                existing_refs.append(existing)

        my_refs: list = []
        if isinstance(reference_latent_cpu, torch.Tensor):
            ref = reference_latent_cpu.to(device=x.device, dtype=x.dtype)
            if ref_scale != 1.0:
                ref = ref * float(ref_scale)
            my_refs.append(ref)

        merged = existing_refs + my_refs
        if len(merged) > 0:
            kwargs["ref_latents"] = merged

        if ref_latents_method_ui == "index_timestep_zero":
            kwargs["ref_latents_method"] = "index_timestep_zero"
        else:
            kwargs["ref_latents_method"] = "offset"

        # Call downstream using keywords so ref_latents is supplied exactly once.
        return executor(
            x,
            timestep,
            context,
            y=y,
            guidance=guidance,
            control=control,
            transformer_options=transformer_options,
            **kwargs,
        )

    return _wrap


class BlueprintPostInputPatch:
    """
    Stock Flux ``post_input`` hook: ``txt`` is already after scene ``txt_norm`` / ``txt_in`` (hidden width).

    Blueprint tensors are raw TE sequence tokens (e.g. [1, N, 12288] for Klein). Last dim must equal
    ``txt_in.in_features`` on the diffusion model; then ``txt_norm`` and ``txt_in`` map them to hidden
    width for concatenation with ``txt``.
    """

    def __init__(
        self,
        model_patcher,
        blueprint_text_tokens_cpu: torch.Tensor,
        keyword_embedding_cpu: Optional[torch.Tensor],
        blueprint_scale: float,
        seed_W: int,
        position: str = "append",
        keyword_position: str = "after_text",
        debug: bool = False,
    ):
        if blueprint_text_tokens_cpu.ndim != 3 or blueprint_text_tokens_cpu.shape[0] != 1:
            raise ValueError("blueprint_text_tokens must be [1, N, D], got {}".format(tuple(blueprint_text_tokens_cpu.shape)))
        self._model_patcher = model_patcher
        self._blueprint_text_tokens_cpu = blueprint_text_tokens_cpu.detach().to(
            dtype=torch.float32, device="cpu"
        ).contiguous()
        self.keyword_embedding_cpu = (
            keyword_embedding_cpu.detach().to("cpu", dtype=torch.float32).reshape(-1).contiguous()
            if isinstance(keyword_embedding_cpu, torch.Tensor)
            else None
        )
        self.blueprint_scale = float(blueprint_scale)
        self.seed_W = int(seed_W)
        self.position = position
        self.keyword_position = keyword_position
        self._debug = bool(debug)
        self._logged_noop = False
        self._logged_first = False
        self._calls = 0
        self._W_kw = None  # type: Optional[torch.Tensor]
        self._cached_d_hidden = None  # type: Optional[int]
        self._cached_kw_in = None  # type: Optional[int]

    def _raw_context_to_txt_hidden(self, raw_seq: torch.Tensor, like_txt: torch.Tensor) -> torch.Tensor:
        """raw_seq [B, N, D_ctx] with D_ctx == txt_in.in_features → txt_norm → txt_in → [B, N, D_hidden]."""
        mp = self._model_patcher
        model = getattr(mp, "model", None)
        if model is None:
            raise RuntimeError("ModelPatcher has no model")
        dm = getattr(model, "diffusion_model", None)
        if dm is None:
            raise RuntimeError("Model has no diffusion_model")
        if not hasattr(dm, "txt_in"):
            raise RuntimeError("diffusion_model has no txt_in; blueprint injection supports Flux-like models only")

        ctx_dim = int(dm.txt_in.weight.shape[1])
        d_bp = int(raw_seq.shape[-1])
        if d_bp != ctx_dim:
            raise RuntimeError(
                "blueprint_text_tokens last dim {} must equal diffusion txt_in.in_features {} "
                "(re-export blueprint with raw TE tokens matching this model).".format(d_bp, ctx_dim)
            )
        x = raw_seq.to(device=like_txt.device, dtype=like_txt.dtype)
        txt_norm = getattr(dm, "txt_norm", None)
        if txt_norm is not None:
            x = txt_norm(x)
        x = dm.txt_in(x)
        return x

    def __call__(self, kwargs):
        self._calls += 1
        img = kwargs["img"]
        txt = kwargs["txt"]
        img_ids = kwargs["img_ids"]
        txt_ids = kwargs["txt_ids"]
        transformer_options = kwargs["transformer_options"]

        if self.blueprint_scale == 0.0:
            if self._debug and not self._logged_noop:
                logger.info("[BlueprintInjector] post_input skipped: blueprint_scale=0")
                self._logged_noop = True
            return kwargs

        L_old = int(txt.shape[1])
        d_hidden = int(txt.shape[-1])

        te = self._blueprint_text_tokens_cpu.to(device=txt.device, dtype=torch.float32)
        B = int(txt.shape[0])
        te_b = te.expand(B, -1, -1)
        try:
            toks = self._raw_context_to_txt_hidden(te_b, txt)
        except Exception as e:
            logger.error("[BlueprintInjector] post_input failed to map blueprint_text_tokens: %s", e)
            raise

        if self.blueprint_scale != 1.0:
            toks = toks * self.blueprint_scale

        kw_tok = None
        use_kw = (
            self.keyword_embedding_cpu is not None
            and str(self.keyword_position).strip().lower() != "disabled"
        )
        if use_kw:
            kw_in = int(self.keyword_embedding_cpu.shape[0])
            if kw_in == d_hidden:
                kw = self.keyword_embedding_cpu
            else:
                if self._W_kw is None or self._cached_d_hidden != d_hidden or self._cached_kw_in != kw_in:
                    self._W_kw = load_or_create_W("W_kw", out_dim=d_hidden, in_dim=kw_in, seed=self.seed_W + 123)
                    self._cached_kw_in = kw_in
                    self._cached_d_hidden = d_hidden
                kw = (self._W_kw @ self.keyword_embedding_cpu).to(dtype=torch.float32, device="cpu").contiguous()

            kw = kw.to(device=txt.device, dtype=txt.dtype)
            if self.blueprint_scale != 1.0:
                kw = kw * self.blueprint_scale
            kw_tok = kw.reshape(1, 1, -1).expand(txt.shape[0], -1, -1)

        if self.position == "prepend":
            if kw_tok is None:
                txt2 = torch.cat([toks, txt], dim=1)
            else:
                txt2 = torch.cat([kw_tok, toks, txt], dim=1)
        else:
            if kw_tok is None:
                txt2 = torch.cat([txt, toks], dim=1)
            else:
                if self.keyword_position == "before_text":
                    txt2 = torch.cat([kw_tok, txt, toks], dim=1)
                else:
                    txt2 = torch.cat([txt, kw_tok, toks], dim=1)

        L_new = int(txt2.shape[1])
        txt_ids2 = _rebuild_txt_ids_for_new_length(txt_ids, L_new)

        if self._debug and not self._logged_first:
            self._logged_first = True
            logger.info(
                "[BlueprintInjector] post_input blueprint_text_tokens call #%s: txt seq %s -> %s; "
                "raw_te_shape=%s d_hidden=%s kw_used=%s position=%s keyword_position=%s",
                self._calls,
                L_old,
                L_new,
                tuple(self._blueprint_text_tokens_cpu.shape),
                d_hidden,
                kw_tok is not None,
                self.position,
                self.keyword_position,
            )
        elif self._debug and self._calls in (10, 100, 500):
            logger.info(
                "[BlueprintInjector] post_input call #%s txt seq %s -> %s",
                self._calls,
                L_old,
                L_new,
            )

        kwargs["txt"] = txt2
        kwargs["txt_ids"] = txt_ids2
        kwargs["img"] = img
        kwargs["img_ids"] = img_ids
        kwargs["transformer_options"] = transformer_options
        return kwargs


class BlueprintInjector:
    CATEGORY = "blueprint"

    _SEED_W_KW = 424242

    @classmethod
    def INPUT_TYPES(cls):
        files = _list_blueprints_input_dir()
        files = [""] + files
        return {
            "required": {
                "model": ("MODEL",),
                "blueprint_path": (files, ),
                "blueprint_scale": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "inject_self_attn": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "self_attn_scale": ("FLOAT", {"default": 0.5, "min": -10.0, "max": 10.0, "step": 0.01, "advanced": True}),
                "bp_tokens_position": (["append", "prepend"], {"default": "append", "advanced": True}),
                "keyword_position": (
                    ["after_text", "before_text", "disabled"],
                    {"default": "after_text", "advanced": True},
                ),
                "ref_scale": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                # When blueprint_path is .index.json: selects entries[] index.
                "reference_frame_index": (
                    "INT",
                    {"default": 0, "min": 0, "max": 8191, "step": 1, "advanced": True},
                ),
                "ref_latents_method": (
                    ["default", "index_timestep_zero"],
                    {"default": "default", "advanced": True},
                ),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "inject"

    def inject(
        self,
        model,
        blueprint_path,
        blueprint_scale=1.0,
        inject_self_attn=False,
        self_attn_scale=0.5,
        bp_tokens_position="append",
        keyword_position="after_text",
        ref_scale=1.0,
        reference_frame_index=0,
        ref_latents_method="default",
        debug=False,
        **_kwargs,
    ):
        resolved = _resolve_blueprint_path(blueprint_path)
        if resolved == "":
            raise RuntimeError("blueprint_path is empty. Select a .index.json from the dropdown.")

        if resolved.lower().endswith(".index.json"):
            # When an index.json is selected, reference_frame_index addresses entries[].
            # This lets the user pick which saved blueprint instance to inject from.
            bp_path, i_raw, i_used, n_entries = _resolve_blueprint_from_index_json_at(
                resolved, entry_index=int(reference_frame_index)
            )
            if bool(debug) and i_raw != i_used:
                logger.info(
                    "[BlueprintInjector] index.json entry_index clamped: %s -> %s (valid 0..%s)",
                    i_raw,
                    i_used,
                    max(0, n_entries - 1),
                )
            resolved = bp_path
        else:
            pass

        m = model.clone()

        bp_ref = load_reference_latent(resolved)  # required; raises if missing

        bp_text = load_blueprint_text_tokens(resolved)

        has_text = bp_text is not None and int(bp_text.shape[1]) >= 1
        use_text = has_text and float(blueprint_scale) > 0.0
        use_ref = True

        # ref is required; load_reference_latent() raises if missing.

        kw = load_keyword_embedding(resolved)
        pos = str(bp_tokens_position).strip().lower()
        if pos not in ("append", "prepend"):
            pos = "append"

        kw_pos = str(keyword_position).strip().lower()
        if kw_pos not in ("after_text", "before_text", "disabled"):
            kw_pos = "after_text"

        ref_ui = str(ref_latents_method).strip().lower()
        if ref_ui not in ("default", "index_timestep_zero"):
            ref_ui = "default"

        dbg = bool(debug)
        if dbg:
            logger.info("[BlueprintInjector] --- debug inject ---")
            logger.info("[BlueprintInjector] blueprint_path resolved=%s", resolved)
            logger.info("[BlueprintInjector] %s", _tensor_debug_line("blueprint_text_tokens", bp_text))
            logger.info("[BlueprintInjector] %s", _tensor_debug_line("reference_latent", bp_ref))
            logger.info("[BlueprintInjector] %s", _tensor_debug_line("keyword_emb", kw))
            logger.info(
                "[BlueprintInjector] has_text=%s use_ref=%s ref_latents_method=%s "
                "reference_frame_index=%s "
                "ref_scale=%s N_tokens=%s blueprint_scale=%s bp_tokens_position=%s keyword_position=%s inject_self_attn=%s",
                use_text,
                use_ref,
                ref_ui,
                int(reference_frame_index),
                float(ref_scale),
                int(bp_text.shape[1]) if has_text else 0,
                float(blueprint_scale),
                pos,
                kw_pos,
                inject_self_attn,
            )

        if use_text:
            post_patch = BlueprintPostInputPatch(
                model_patcher=m,
                blueprint_text_tokens_cpu=bp_text,
                keyword_embedding_cpu=kw,
                blueprint_scale=float(blueprint_scale),
                seed_W=int(self._SEED_W_KW),
                position=pos,
                keyword_position=kw_pos,
                debug=dbg,
            )
            m.set_model_post_input_patch(post_patch)
            if dbg:
                logger.info("[BlueprintInjector] set_model_post_input_patch registered.")
        elif dbg:
            logger.info("[BlueprintInjector] post_input skipped (no blueprint_text_tokens).")

        if use_ref:
            rs = float(ref_scale)
            if rs < 0.0:
                rs = 0.0
            if rs > 3.0:
                rs = 3.0
            wrap = _blueprint_reference_latent_diffusion_wrapper(
                bp_ref,
                ref_ui,
                rs,
            )
            m.add_wrapper(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL, wrap)
            if dbg:
                logger.info(
                    "[BlueprintInjector] DIFFUSION_MODEL wrapper registered for reference_latent "
                    "(shape=%s, method=%s, ref_scale=%s).",
                    tuple(bp_ref.shape),
                    ref_ui,
                    rs,
                )

        if inject_self_attn and dbg:
            logger.info(
                "[BlueprintInjector] inject_self_attn=True ignored for Flux.2 joint attention "
                "(self_attn_scale=%s).",
                float(self_attn_scale),
            )

        return (m,)
