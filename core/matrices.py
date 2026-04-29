import os

import torch

from .deps import require_safetensors
from .paths import matrices_dir


def load_or_create_R(name: str, out_dim: int, in_dim: int, seed: int) -> torch.Tensor:
    """
    Fixed random projection matrix cached on disk.
    Stored under matrices/ as safetensors with key 'R'.
    """
    safetensors = require_safetensors()

    path = os.path.join(matrices_dir(), "{}_{}x{}_seed{}.safetensors".format(name, out_dim, in_dim, seed))
    if os.path.exists(path):
        t = safetensors.torch.load_file(path, device="cpu").get("R")
        if isinstance(t, torch.Tensor) and tuple(t.shape) == (out_dim, in_dim):
            return t.to(dtype=torch.float32, device="cpu").contiguous()

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0xFFFFFFFF)
    R = torch.randn((out_dim, in_dim), generator=g, dtype=torch.float32, device="cpu") / (in_dim ** 0.5)
    safetensors.torch.save_file({"R": R.contiguous()}, path, metadata={"seed": str(seed), "name": name})
    return R


def load_or_create_W(name: str, out_dim: int, in_dim: int, seed: int) -> torch.Tensor:
    """
    Fixed random projection matrix cached on disk.
    Stored under matrices/ as safetensors with key 'W'.
    """
    safetensors = require_safetensors()

    path = os.path.join(matrices_dir(), "{}_{}x{}_seed{}.safetensors".format(name, out_dim, in_dim, seed))
    if os.path.exists(path):
        t = safetensors.torch.load_file(path, device="cpu").get("W")
        if isinstance(t, torch.Tensor) and tuple(t.shape) == (out_dim, in_dim):
            return t.to(dtype=torch.float32, device="cpu").contiguous()

    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed) & 0xFFFFFFFF)
    W = torch.randn((out_dim, in_dim), generator=g, dtype=torch.float32, device="cpu") / (in_dim ** 0.5)
    safetensors.torch.save_file({"W": W.contiguous()}, path, metadata={"seed": str(seed), "name": name})
    return W

