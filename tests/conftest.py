from __future__ import annotations

import pytest


@pytest.fixture
def engine():
    from fakes import FakeEngine

    return FakeEngine(output="Let me think.\nDone.\n</think>\n\nThe answer is 391.")


@pytest.fixture
def settings():
    from openai_maple.config import Settings

    return Settings(api_key=None, max_new_tokens=64, allowed_origins=[])


@pytest.fixture
def client(settings, engine):
    from fastapi.testclient import TestClient

    from openai_maple.server import create_app

    app = create_app(settings=settings, engine=engine)
    with TestClient(app) as test_client:
        test_client.engine = engine  # type: ignore[attr-defined]
        yield test_client
