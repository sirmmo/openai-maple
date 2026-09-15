# Development

```bash
git clone https://github.com/sirmmo/openai-maple
cd openai-maple
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e '.[test]'
```

## Tests

Three tiers, selected with pytest markers:

| Command | What runs | Needs |
| --- | --- | --- |
| `pytest` | offline suite: fake engine behind the HTTP surface, SSE and parser tests, OpenAI SDK round trip, numerical checks of the attention shim and the ternary packing | nothing |
| `pytest -m network` | builds a two-layer random-init Maple from the HF-hosted custom code and runs real `generate()` through the engine: streaming, stop, cancel, seeds, packing | a few MB from HuggingFace, cached |
| `OPENAI_MAPLE_URL=http://127.0.0.1:8000 pytest -m live` | end-to-end against a running server with the real weights | a server; minutes per call on CPU |

CI runs the first two on Python 3.10 and 3.12, plus `ruff check` and
`ruff format --check`.

## Iterating on the generation path

Do not reload 40 GB to debug `generate()`. The `tiny_model_dir` fixture in
`tests/test_engine_tiny.py` writes a 2-layer, 8-expert, 64-wide Maple (about
90 MB) that goes through the same custom `modeling_maple.py`, the shim, the
`fa3.py` patch and the `DynamicCache` path. Both upstream bugs reproduce on
it in seconds.

## Releases

- Pushes to `main` publish `ghcr.io/sirmmo/openai-maple:latest` (CPU,
  amd64 + arm64), `:latest-cuda` and immutable `sha-*` tags
  (`.github/workflows/docker.yml`).
- A `vX.Y.Z` tag additionally builds the sdist and wheel, publishes to PyPI
  through trusted publishing, creates a GitHub release, and tags the images
  `X.Y.Z`, `X.Y` and their `-cuda` twins (`.github/workflows/release.yml`).
  `workflow_dispatch` on that workflow is a dry run: it builds and checks the
  distributions and stops.
- This site is built with MkDocs Material from `docs/` on every push that
  touches it (`.github/workflows/docs.yml`).

Bump `version` in `pyproject.toml` and `openai_maple/__init__.py`, add a
`CHANGELOG.md` entry, tag, push.
