"""Regression: JointConstellation must fall back to CPU when device='cuda' is requested but CUDA is
absent — never assign a dead 'cuda' device string that torch cannot place tensors on (lifeguard, 2026-07-01)."""

from __future__ import annotations

import torch

from qig_coordizer import FisherCoordizer
from qig_studio.constellation.joint_trainer import JointConstellation
from qig_studio.development import PROTOMAP_ORDER

_COORDIZER = "../qig-coordizer/checkpoints/coordizer_20260701_8k_v1.json"


def _all_tensor_devices(obj, _seen=None, _depth=0):
    """Every torch device found under obj (modules + tensors), recursively (bounded)."""
    if _seen is None:
        _seen = set()
    out: set[str] = set()
    if _depth > 4 or id(obj) in _seen:
        return out
    _seen.add(id(obj))
    if isinstance(obj, torch.nn.Module):
        for p in obj.parameters():
            out.add(str(p.device))
        return out
    for attr in vars(obj).values() if hasattr(obj, "__dict__") else []:
        if isinstance(attr, torch.Tensor):
            out.add(str(attr.device))
        elif isinstance(attr, torch.nn.Module):
            out |= _all_tensor_devices(attr, _seen, _depth + 1)
        elif hasattr(attr, "__dict__"):
            out |= _all_tensor_devices(attr, _seen, _depth + 1)
    return out


def test_cuda_requested_but_absent_falls_back_to_cpu(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cz = FisherCoordizer.load(_COORDIZER)
    # device='cuda' with CUDA unavailable: must build on cpu, NOT raise a placement error.
    jc = JointConstellation(list(PROTOMAP_ORDER)[:2], num_layers=1, coordizer=cz, device="cuda")
    devs = _all_tensor_devices(jc.central)
    assert devs, "no torch tensors found on central to check device"
    assert all("cuda" not in d for d in devs), f"cuda-absent fallback failed: central tensors on {devs}"
