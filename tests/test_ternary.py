from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from openai_maple.ternary import TernaryMLP, pack_ternary  # noqa: E402


def _ternary(rows: int, cols: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(-1, 2, (rows, cols), generator=g).float()
    scale = torch.rand(rows, 1, generator=g) * 0.1 + 0.01
    return (signs * scale).to(torch.bfloat16)


def test_pack_round_trips_exactly():
    w = _ternary(64, 128)
    q, s = pack_ternary(w)
    assert q.dtype == torch.int8 and s.dtype == torch.float32
    assert torch.equal((q.float() * s[:, None]).to(torch.bfloat16), w)


def test_pack_handles_all_zero_rows():
    w = _ternary(8, 16)
    w[3] = 0
    q, s = pack_ternary(w)
    assert s[3] == 0 and torch.all(q[3] == 0)


def test_pack_rejects_non_ternary():
    assert pack_ternary(torch.randn(8, 16, dtype=torch.bfloat16)) is None
    assert pack_ternary(torch.ones(8)) is None


def test_ternary_mlp_matches_reference_forward():
    gate, up, down = _ternary(32, 64, 1), _ternary(32, 64, 2), _ternary(64, 32, 3)
    act = torch.nn.functional.silu
    mlp = TernaryMLP(pack_ternary(gate), pack_ternary(up), pack_ternary(down), act)
    x = torch.randn(5, 64)

    def reference(x):
        g = torch.nn.functional.linear(x, gate.float())
        u = torch.nn.functional.linear(x, up.float())
        return torch.nn.functional.linear(
            act(torch.clamp(g, max=7.0)) * torch.clamp(u, -7.0, 7.0), down.float()
        )

    torch.testing.assert_close(mlp(x), reference(x), atol=1e-5, rtol=1e-5)
    assert (
        sum(b.numel() * b.element_size() for b in mlp.buffers())
        < (gate.numel() + up.numel() + down.numel()) * 2
    )
