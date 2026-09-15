"""CPU fast path: pack Maple's ternary expert weights as int8 + per-row scale.

Every expert projection in the checkpoint is row-wise ternary: each row holds
``{-s, 0, +s}`` for one bf16 scale ``s``. Storing ``sign`` as int8 and ``s`` as
float32 reproduces the bf16 tensor bit-for-bit while halving memory (the
20B-parameter expert bank drops from 40GB to ~19GB).

The bigger win is speed. PyTorch's bf16 GEMV on CPUs without AVX-512-BF16/AMX
is ~100x slower than fp32 at the expert shapes (512x2048), and every token
runs 24 layers x 8 experts x 3 projections through it. Dequantising the
active expert into fp32 on the fly costs a fraction of that.

The remaining ~0.9B non-expert parameters (attention, router, embeddings,
lm_head, norms) are cast to fp32, so hidden states stay fp32 end to end.
"""

from __future__ import annotations

import logging
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

log = logging.getLogger(__name__)


def pack_ternary(weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return ``(int8 signs, float32 row scales)`` or ``None`` if not exactly ternary."""
    if weight.dim() != 2 or not weight.is_floating_point():
        return None
    w = weight.detach()
    scale = w.abs().amax(dim=1)  # exact in the source dtype
    q = torch.sign(w).to(torch.int8)
    # Row-wise ternary means sign * rowmax reproduces every element exactly.
    if not torch.equal(q.to(w.dtype) * scale[:, None], w):
        return None
    return q, scale.float()


class TernaryMLP(nn.Module):
    """Drop-in replacement for one Maple expert (``MapleMLP``).

    Reproduces the original forward exactly::

        down( act(clamp(gate(x), max=7)) * clamp(up(x), -7, 7) )
    """

    def __init__(
        self,
        gate: tuple[torch.Tensor, torch.Tensor],
        up: tuple[torch.Tensor, torch.Tensor],
        down: tuple[torch.Tensor, torch.Tensor],
        act_fn: Any,
    ) -> None:
        super().__init__()
        self.register_buffer("gate_q", gate[0], persistent=False)
        self.register_buffer("gate_s", gate[1], persistent=False)
        self.register_buffer("up_q", up[0], persistent=False)
        self.register_buffer("up_s", up[1], persistent=False)
        self.register_buffer("down_q", down[0], persistent=False)
        self.register_buffer("down_s", down[1], persistent=False)
        self.act_fn = act_fn

    @staticmethod
    def _dequant(q: torch.Tensor, s: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        return q.to(dtype) * s.to(dtype)[:, None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        gate = F.linear(x, self._dequant(self.gate_q, self.gate_s, dt))
        up = F.linear(x, self._dequant(self.up_q, self.up_s, dt))
        h = self.act_fn(torch.clamp(gate, max=7.0)) * torch.clamp(up, min=-7.0, max=7.0)
        return F.linear(h, self._dequant(self.down_q, self.down_s, dt))


def pack_experts(model: Any) -> dict[str, int]:
    """Replace every ternary expert in ``model`` with a :class:`TernaryMLP` and
    cast the remaining parameters to float32. Returns counts for logging."""
    packed = kept = 0
    layers = model.model.layers
    for layer in layers:
        moe = getattr(layer, "mlp", None)
        experts = getattr(moe, "experts", None)
        if experts is None:
            continue
        for i in range(len(experts)):
            expert = experts[i]
            parts = [
                pack_ternary(getattr(expert, name).weight) for name in ("gate_proj", "up_proj", "down_proj")
            ]
            if any(p is None for p in parts):
                kept += 1
                continue
            experts[i] = TernaryMLP(parts[0], parts[1], parts[2], expert.act_fn)  # type: ignore[arg-type]
            packed += 1
            del expert
    # Cast what is left (attention, router, embeddings, lm_head, norms) to fp32.
    # ``.float()`` only touches floating-point tensors; int8 buffers stay put.
    model.float()
    log.info("packed %d experts to int8 ternary, %d kept as fp32", packed, kept)
    return {"packed": packed, "kept": kept}
