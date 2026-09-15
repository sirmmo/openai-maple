"""End-to-end engine tests on a tiny random-init Maple built from the real
custom modeling code. Needs the code/tokenizer files from HuggingFace (a few
MB, cached) but not the 40GB of weights.

Run with: pytest -m network
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import threading

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

pytestmark = pytest.mark.network

REPO = "deepgrove/maple-preview"


@pytest.fixture(scope="module")
def tiny_model_dir(tmp_path_factory):
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError, LocalEntryNotFoundError

    from openai_maple import flash_attn_shim

    flash_attn_shim.install()
    try:
        src = snapshot_download(REPO, allow_patterns=["*.py", "*.json", "*.txt", "*.jinja"])
    except (HfHubHTTPError, LocalEntryNotFoundError, OSError) as exc:
        pytest.skip(f"cannot fetch model code: {exc}")

    out = tmp_path_factory.mktemp("tiny-maple")
    for f in glob.glob(f"{src}/*"):
        name = os.path.basename(f)
        if os.path.isfile(f) and name != "config.json" and not name.endswith((".safetensors", ".index.json")):
            shutil.copy(f, out)
    with open(f"{src}/config.json") as fh:
        cfg = json.load(fh)
    cfg.update(
        hidden_size=64,
        num_hidden_layers=2,
        layer_types=["sliding_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=8,
        num_experts_per_tok=2,
        sliding_window=4,
    )
    with open(out / "config.json", "w") as fh:
        json.dump(cfg, fh)

    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(out, trust_remote_code=True)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True, dtype=torch.float32)
    model.save_pretrained(out)
    return str(out)


@pytest.fixture(scope="module")
def engine(tiny_model_dir):
    from openai_maple.engine import MapleEngine

    eng = MapleEngine(tiny_model_dir, device="cpu", dtype="float32", torch_threads=4)
    eng.start()
    return eng


def test_start_reports_backend(engine):
    assert engine.ready
    assert engine.device == "cpu"
    assert "shim" in engine.attention_backend or engine.attention_backend == "flash_attn"


def test_chat_template_opens_think_block(engine):
    text = engine.apply_chat_template([{"role": "user", "content": "hi"}])
    assert text.endswith("<|im_start|>assistant\n<think>\n")
    text = engine.apply_chat_template([{"role": "user", "content": "hi"}], enable_thinking=False)
    assert text.endswith("<think>\n\n</think>\n\n")


def test_stream_yields_pieces_and_length_finish(engine):
    from openai_maple.engine import GenerationParams

    ids = engine.encode("hello there")
    pieces = list(engine.stream(ids, GenerationParams(max_new_tokens=6, temperature=0.0)))
    assert pieces[-1].finish_reason == "length"
    assert pieces[-1].completion_tokens == 6
    assert "".join(p.text for p in pieces)  # random weights still produce text


def test_greedy_is_deterministic_and_matches_full_forward(engine):
    from openai_maple.engine import GenerationParams

    ids = engine.encode("one two three four five six seven")
    a = "".join(p.text for p in engine.stream(ids, GenerationParams(max_new_tokens=8, temperature=0.0)))
    b = "".join(p.text for p in engine.stream(ids, GenerationParams(max_new_tokens=8, temperature=0.0)))
    assert a == b

    # Teacher-force the generated ids through a cache-less, mask-less forward
    # (the dense attention path) and check every greedy pick agrees.
    tok = engine.tokenizer
    gen_ids = ids + tok(a, add_special_tokens=False)["input_ids"]
    with torch.inference_mode():
        logits = engine.model(input_ids=torch.tensor([gen_ids]), use_cache=False).logits[0]
    pred = logits[len(ids) - 1 : -1].argmax(-1).tolist()
    assert pred == gen_ids[len(ids) :]


def test_seed_makes_sampling_reproducible(engine):
    from openai_maple.engine import GenerationParams

    ids = engine.encode("hello")
    p = GenerationParams(max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0, seed=42)
    a = "".join(x.text for x in engine.stream(ids, p))
    b = "".join(x.text for x in engine.stream(ids, p))
    assert a == b


def test_cancel_stops_generation(engine):
    from openai_maple.engine import GenerationParams

    ids = engine.encode("hello")
    cancel = threading.Event()
    seen = 0
    finish = None
    for piece in engine.stream(ids, GenerationParams(max_new_tokens=200, temperature=0.0), cancel):
        seen += 1
        if seen == 3:
            cancel.set()
        finish = piece.finish_reason
    assert finish == "cancelled"
    assert seen < 100


def test_prompt_too_long(engine):
    from openai_maple.engine import PromptTooLong

    engine.max_prompt_tokens = 3
    try:
        with pytest.raises(PromptTooLong):
            engine.encode("this prompt is definitely more than three tokens long")
    finally:
        engine.max_prompt_tokens = 65536


def test_pack_experts_on_tiny_model(tiny_model_dir):
    """Random-init experts are not ternary, so packing must keep them and
    still leave a working fp32 model; forcing ternary weights must pack."""
    from openai_maple import flash_attn_shim
    from openai_maple.engine import _patch_query_length_bug
    from openai_maple.ternary import TernaryMLP, pack_experts

    flash_attn_shim.install()
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, trust_remote_code=True, dtype=torch.bfloat16)
    _patch_query_length_bug(model)
    ids = torch.randint(0, 1000, (1, 9))
    with torch.inference_mode():
        before = model(input_ids=ids, use_cache=False).logits.float()

    # Make layer 0's experts ternary so both branches run.
    for expert in model.model.layers[0].mlp.experts:
        for name in ("gate_proj", "up_proj", "down_proj"):
            w = getattr(expert, name).weight
            w.data = (torch.sign(w.float()) * w.float().abs().amax(dim=1, keepdim=True)).to(torch.bfloat16)
    with torch.inference_mode():
        ternary_before = model(input_ids=ids, use_cache=False).logits.float()

    counts = pack_experts(model)
    assert counts["packed"] == len(model.model.layers[0].mlp.experts)
    assert counts["kept"] == len(model.model.layers[1].mlp.experts)
    assert isinstance(model.model.layers[0].mlp.experts[0], TernaryMLP)
    assert model.lm_head.weight.dtype == torch.float32
    with torch.inference_mode():
        after = model(input_ids=ids, use_cache=False).logits
    # fp32 compute vs the bf16 original: same weights, only rounding differs.
    assert torch.allclose(after, ternary_before, atol=0.5, rtol=0.05)
    assert not torch.allclose(after, before, atol=1e-6)
