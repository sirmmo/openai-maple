"""Numerical checks of the pure-torch flash_attn shim against a naive reference."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openai_maple import flash_attn_shim as shim  # noqa: E402


def naive(q, k, v, *, causal, window_left, window_right, scale=None):
    """Explicit O(L^2) reference with FlashAttention's bottom-right alignment."""
    bsz, len_q, heads_q, head_dim = q.shape
    len_k, heads_kv = k.shape[1], k.shape[2]
    scale = scale or head_dim**-0.5
    rep = heads_q // heads_kv
    kk = k.repeat_interleave(rep, dim=2).float()
    vv = v.repeat_interleave(rep, dim=2).float()
    scores = torch.einsum("bqhd,bkhd->bhqk", q.float(), kk) * scale
    qi = torch.arange(len_q)[:, None] + (len_k - len_q)
    kj = torch.arange(len_k)[None, :]
    allowed = torch.ones(len_q, len_k, dtype=torch.bool)
    if causal:
        allowed &= kj <= qi
    if window_left >= 0:
        allowed &= kj >= qi - window_left
    if window_right >= 0:
        allowed &= kj <= qi + window_right
    scores = scores.masked_fill(~allowed, float("-inf"))
    probs = scores.softmax(-1)
    return torch.einsum("bhqk,bkhd->bqhd", probs, vv).to(q.dtype)


@pytest.mark.parametrize("len_q,len_k", [(7, 7), (1, 40), (5, 40), (1500, 1500), (3, 1300)])
@pytest.mark.parametrize("window", [(-1, -1), (4, 0), (512, 0)])
@pytest.mark.parametrize("causal", [True, False])
def test_dense_matches_reference(len_q, len_k, window, causal):
    torch.manual_seed(0)
    q = torch.randn(2, len_q, 8, 16)
    k = torch.randn(2, len_k, 2, 16)
    v = torch.randn(2, len_k, 2, 16)
    if not causal and window == (4, 0) and len_q != len_k:
        pytest.skip("non-causal windowed with mismatched lengths is not a FlashAttention use case")
    got = shim.flash_attn_func(q, k, v, causal=causal, window_size=window)
    want = naive(q, k, v, causal=causal, window_left=window[0], window_right=window[1])
    torch.testing.assert_close(got, want, atol=1e-4, rtol=1e-4)


def test_bf16_inputs_return_bf16():
    q = torch.randn(1, 9, 4, 8, dtype=torch.bfloat16)
    k = torch.randn(1, 9, 1, 8, dtype=torch.bfloat16)
    v = torch.randn(1, 9, 1, 8, dtype=torch.bfloat16)
    out = shim.flash_attn_func(q, k, v, causal=True)
    assert out.dtype == torch.bfloat16
    assert out.shape == (1, 9, 4, 8)


def test_varlen_matches_per_sequence_dense():
    torch.manual_seed(1)
    lens = [5, 1, 9]
    q = torch.randn(sum(lens), 4, 8)
    k = torch.randn(sum(lens), 2, 8)
    v = torch.randn(sum(lens), 2, 8)
    cu = torch.tensor([0, 5, 6, 15], dtype=torch.int32)
    got = shim.flash_attn_varlen_func(q, k, v, cu, cu, 9, 9, causal=True, window_size=(3, 0))
    start = 0
    for n in lens:
        want = naive(
            q[start : start + n][None],
            k[start : start + n][None],
            v[start : start + n][None],
            causal=True,
            window_left=3,
            window_right=0,
        )[0]
        torch.testing.assert_close(got[start : start + n], want, atol=1e-4, rtol=1e-4)
        start += n


def test_padding_round_trip():
    x = torch.arange(2 * 4 * 3, dtype=torch.float32).view(2, 4, 3)
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    flat, indices, cu, max_len, seqlens = shim.unpad_input(x, mask)
    assert flat.shape == (5, 3)
    assert cu.tolist() == [0, 3, 5]
    assert max_len == 3
    back = shim.pad_input(flat, indices, 2, 4)
    assert torch.equal(back * mask[..., None], x * mask[..., None])


def test_install_registers_package(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "flash_attn", raising=False)
    monkeypatch.delitem(sys.modules, "flash_attn.bert_padding", raising=False)
    assert shim.install(force=True) is True
    import flash_attn  # noqa: F401
    from flash_attn.bert_padding import pad_input  # noqa: F401

    assert flash_attn.flash_attn_func is shim.flash_attn_func
