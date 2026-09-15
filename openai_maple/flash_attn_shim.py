"""Pure-torch stand-in for the ``flash_attn`` package.

The Transformers implementation shipped with deepgrove/maple-preview imports
``flash_attn`` unconditionally (see ``fa3.py`` in the model repo), which only
builds for CUDA. This module reproduces the three entry points that file uses
-- ``flash_attn_func``, ``flash_attn_varlen_func`` and ``flash_attn.bert_padding``
-- on top of ``torch.nn.functional.scaled_dot_product_attention`` so the same
model code runs on CPU, MPS, or a CUDA box without the FlashAttention wheel.

Semantics mirror FlashAttention 2:

* tensors are ``(batch, seqlen, heads, head_dim)``, not head-first;
* grouped-query attention is implicit (``heads_q`` a multiple of ``heads_kv``);
* ``causal=True`` uses *bottom-right* alignment, so a single decode query with
  a long KV cache attends to every key;
* ``window_size=(left, right)`` restricts key ``j`` to ``i-left <= j <= i+right``
  in the same aligned coordinates, ``-1`` meaning unbounded.

Queries are processed in chunks so the boolean mask never grows past
``chunk x (chunk + window)`` on sliding-window layers, and full-attention prefill
uses ``is_causal`` with no materialised mask at all.

Call :func:`install` before ``AutoModelForCausalLM.from_pretrained``; it is a
no-op when a real ``flash_attn`` (or FlashAttention 3's ``flash_attn_interface``)
is importable.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
import types

import torch
import torch.nn.functional as F

_CHUNK = 1024


def _attend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    scale: float | None,
    causal: bool,
    window_left: int,
    window_right: int,
) -> torch.Tensor:
    """Attention over ``(B, L, H, D)`` inputs; returns ``(B, Lq, Hq, D)``."""
    bsz, len_q, heads_q, head_dim = q.shape
    len_k, heads_kv = k.shape[1], k.shape[2]
    if scale is None:
        scale = head_dim**-0.5

    compute_dtype = q.dtype
    # bf16 SDPA on CPU is slow and imprecise on older CPUs; attention is a
    # small share of this model's FLOPs, so upcast there.
    if q.device.type == "cpu" and q.dtype in (torch.bfloat16, torch.float16):
        compute_dtype = torch.float32

    qh = q.transpose(1, 2).to(compute_dtype)  # B, Hq, Lq, D
    kh = k.transpose(1, 2).to(compute_dtype)
    vh = v.transpose(1, 2).to(compute_dtype)
    if heads_kv != heads_q:
        rep = heads_q // heads_kv
        kh = kh.repeat_interleave(rep, dim=1)
        vh = vh.repeat_interleave(rep, dim=1)

    offset = len_k - len_q  # bottom-right alignment
    unbounded = window_left < 0 and window_right < 0

    if not causal and unbounded:
        out = F.scaled_dot_product_attention(qh, kh, vh, scale=scale)
        return out.transpose(1, 2).to(q.dtype)

    if causal and unbounded and offset == 0:
        out = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True, scale=scale)
        return out.transpose(1, 2).to(q.dtype)

    device = q.device
    pieces = []
    for start in range(0, len_q, _CHUNK):
        end = min(len_q, start + _CHUNK)
        q_pos = torch.arange(start, end, device=device) + offset  # aligned query index

        k_lo = 0
        if window_left >= 0:
            k_lo = max(0, start + offset - window_left)
        k_hi = len_k
        if causal:
            k_hi = min(k_hi, end + offset)
        elif window_right >= 0:
            k_hi = min(k_hi, end + offset + window_right)
        if causal and window_right >= 0:
            k_hi = min(k_hi, end + offset)
        k_lo = min(k_lo, k_hi)  # degenerate guard

        k_pos = torch.arange(k_lo, k_hi, device=device)
        allowed = torch.ones((end - start, k_hi - k_lo), dtype=torch.bool, device=device)
        if causal:
            allowed &= k_pos[None, :] <= q_pos[:, None]
        if window_left >= 0:
            allowed &= k_pos[None, :] >= (q_pos[:, None] - window_left)
        if window_right >= 0:
            allowed &= k_pos[None, :] <= (q_pos[:, None] + window_right)
        # A row with no permitted key would produce NaN; FlashAttention returns
        # zeros there. Only reachable when len_q > len_k, which generation
        # never does, but keep the output finite regardless.
        empty_rows = ~allowed.any(dim=1)
        if bool(empty_rows.any()):
            allowed[empty_rows, 0] = True

        out = F.scaled_dot_product_attention(
            qh[:, :, start:end],
            kh[:, :, k_lo:k_hi],
            vh[:, :, k_lo:k_hi],
            attn_mask=allowed,
            scale=scale,
        )
        if bool(empty_rows.any()):
            out[:, :, empty_rows] = 0
        pieces.append(out)

    out = pieces[0] if len(pieces) == 1 else torch.cat(pieces, dim=2)
    return out.transpose(1, 2).to(q.dtype)


def flash_attn_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
):
    if softcap:
        raise NotImplementedError("softcap is not supported by the CPU attention shim")
    if alibi_slopes is not None:
        raise NotImplementedError("alibi_slopes is not supported by the CPU attention shim")
    out = _attend(
        q, k, v, scale=softmax_scale, causal=causal, window_left=window_size[0], window_right=window_size[1]
    )
    return (out, None, None) if return_attn_probs else out


def flash_attn_varlen_func(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    dropout_p: float = 0.0,
    softmax_scale: float | None = None,
    causal: bool = False,
    window_size: tuple[int, int] = (-1, -1),
    softcap: float = 0.0,
    alibi_slopes=None,
    deterministic: bool = False,
    return_attn_probs: bool = False,
    block_table=None,
):
    """Packed-sequence variant: ``q`` is ``(total_q, H, D)`` and each segment
    ``cu_seqlens[i]:cu_seqlens[i+1]`` is one sequence."""
    if softcap:
        raise NotImplementedError("softcap is not supported by the CPU attention shim")
    starts_q = cu_seqlens_q.tolist()
    starts_k = cu_seqlens_k.tolist()
    pieces = []
    for i in range(len(starts_q) - 1):
        qs, qe = starts_q[i], starts_q[i + 1]
        ks, ke = starts_k[i], starts_k[i + 1]
        if qe == qs:
            continue
        out = _attend(
            q[qs:qe].unsqueeze(0),
            k[ks:ke].unsqueeze(0),
            v[ks:ke].unsqueeze(0),
            scale=softmax_scale,
            causal=causal,
            window_left=window_size[0],
            window_right=window_size[1],
        )
        pieces.append(out.squeeze(0))
    out = torch.cat(pieces, dim=0) if pieces else q.new_empty((0, q.shape[1], q.shape[2]))
    return (out, None, None) if return_attn_probs else out


# --- flash_attn.bert_padding -------------------------------------------------


def index_first_axis(x: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return x[indices]


def index_put_first_axis(values: torch.Tensor, indices: torch.Tensor, first_axis_dim: int) -> torch.Tensor:
    out = torch.zeros((first_axis_dim, *values.shape[1:]), dtype=values.dtype, device=values.device)
    out[indices] = values
    return out


def unpad_input(hidden_states: torch.Tensor, attention_mask: torch.Tensor, unused_mask=None):
    seqlens = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = int(seqlens.max().item()) if seqlens.numel() else 0
    cu_seqlens = F.pad(torch.cumsum(seqlens, dim=0, dtype=torch.int32), (1, 0))
    flat = hidden_states.reshape(-1, *hidden_states.shape[2:])
    return index_first_axis(flat, indices), indices, cu_seqlens, max_seqlen, seqlens


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch: int, seqlen: int) -> torch.Tensor:
    out = index_put_first_axis(hidden_states, indices, batch * seqlen)
    return out.view(batch, seqlen, *hidden_states.shape[1:])


# --- registration ------------------------------------------------------------


def real_flash_attn_available() -> bool:
    for name in ("flash_attn_interface", "flash_attn"):
        try:
            if importlib.util.find_spec(name) is not None:
                return True
        except (ImportError, ValueError):
            continue
    return False


def install(force: bool = False) -> bool:
    """Register the shim as ``flash_attn`` in ``sys.modules``.

    Returns ``True`` when the shim was installed, ``False`` when a real
    FlashAttention build is present and left untouched.
    """
    if not force and real_flash_attn_available():
        return False
    if "flash_attn" in sys.modules and getattr(sys.modules["flash_attn"], "__maple_shim__", False):
        return True

    pkg = types.ModuleType("flash_attn")
    pkg.__maple_shim__ = True  # type: ignore[attr-defined]
    pkg.__path__ = []  # type: ignore[attr-defined]  # mark as package
    # transformers probes ``importlib.util.find_spec("flash_attn")`` at import
    # time, which raises on a module whose __spec__ is None. A spec without a
    # distribution behind it makes that probe answer "not installed", so
    # transformers keeps its own SDPA paths and only the Maple code sees the shim.
    pkg.__spec__ = importlib.machinery.ModuleSpec("flash_attn", None, is_package=True)
    pkg.__version__ = "0.0.0+maple-shim"  # type: ignore[attr-defined]
    pkg.flash_attn_func = flash_attn_func  # type: ignore[attr-defined]
    pkg.flash_attn_varlen_func = flash_attn_varlen_func  # type: ignore[attr-defined]

    padding = types.ModuleType("flash_attn.bert_padding")
    padding.__spec__ = importlib.machinery.ModuleSpec("flash_attn.bert_padding", None)
    padding.index_first_axis = index_first_axis  # type: ignore[attr-defined]
    padding.index_put_first_axis = index_put_first_axis  # type: ignore[attr-defined]
    padding.unpad_input = unpad_input  # type: ignore[attr-defined]
    padding.pad_input = pad_input  # type: ignore[attr-defined]
    pkg.bert_padding = padding  # type: ignore[attr-defined]

    sys.modules["flash_attn"] = pkg
    sys.modules["flash_attn.bert_padding"] = padding
    return True
