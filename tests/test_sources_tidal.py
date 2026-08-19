import asyncio

import pytest

from salmon.errors import ScrapeError
from salmon.sources.base import BaseScraper
from salmon.sources.tidal import (
    _MAX_RATE_LIMIT_RETRIES,
    TidalBase,
    _parse_retry_after,
    _TidalRateLimitError,
)


@pytest.fixture(autouse=True)
def _reset_tidal_rate_limit_state():
    TidalBase._cooldown_until = 0.0
    TidalBase._access_token = None
    TidalBase._token_expiry = 0.0
    yield
    TidalBase._cooldown_until = 0.0


def test_parse_retry_after_seconds() -> None:
    assert _parse_retry_after("2.5") == 2.5


def test_parse_retry_after_caps_and_rejects_invalid() -> None:
    assert _parse_retry_after("120") == 60.0
    assert _parse_retry_after("-3") == 0.0
    assert _parse_retry_after("Wed, 21 Oct 2015 07:28:00 GMT") is None
    assert _parse_retry_after(None) is None


def test_get_json_waits_and_retries_after_429(monkeypatch) -> None:
    calls = 0
    sleeps: list[float] = []

    async def fake_token(cls) -> str:
        return "tok"

    async def fake_parent_get_json(self, url, params=None, headers=None) -> dict:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise _TidalRateLimitError(0.25)
        return {"ok": True}

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(TidalBase, "_ensure_token", classmethod(fake_token))
    monkeypatch.setattr(BaseScraper, "get_json", fake_parent_get_json)
    monkeypatch.setattr("salmon.sources.tidal.asyncio.sleep", fake_sleep)

    result = asyncio.run(TidalBase().get_json("/albums/1"))

    assert result == {"ok": True}
    assert calls == 2
    assert sleeps
    assert sleeps[0] == pytest.approx(0.25, abs=0.05)


def test_get_json_raises_after_retries_exhausted(monkeypatch) -> None:
    async def fake_token(cls) -> str:
        return "tok"

    async def fake_parent_get_json(self, url, params=None, headers=None) -> dict:
        raise _TidalRateLimitError(0.01)

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(TidalBase, "_ensure_token", classmethod(fake_token))
    monkeypatch.setattr(BaseScraper, "get_json", fake_parent_get_json)
    monkeypatch.setattr("salmon.sources.tidal.asyncio.sleep", fake_sleep)

    with pytest.raises(ScrapeError, match="rate limit persisted"):
        asyncio.run(TidalBase().get_json("/albums/1"))


def test_shared_cooldown_blocks_later_requests(monkeypatch) -> None:
    parent_calls = 0
    sleeps: list[float] = []

    async def fake_token(cls) -> str:
        return "tok"

    async def fake_parent_get_json(self, url, params=None, headers=None) -> dict:
        nonlocal parent_calls
        parent_calls += 1
        if parent_calls == 1:
            raise _TidalRateLimitError(1.5)
        return {"n": parent_calls}

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)

    monkeypatch.setattr(TidalBase, "_ensure_token", classmethod(fake_token))
    monkeypatch.setattr(BaseScraper, "get_json", fake_parent_get_json)
    monkeypatch.setattr("salmon.sources.tidal.asyncio.sleep", fake_sleep)

    scraper = TidalBase()
    first = asyncio.run(scraper.get_json("/albums/1"))
    second = asyncio.run(scraper.get_json("/albums/2"))

    assert first == {"n": 2}
    assert second == {"n": 3}
    assert any(s == pytest.approx(1.5, abs=0.05) for s in sleeps)
    assert parent_calls == 3


def test_rate_limit_retry_count_matches_constant(monkeypatch) -> None:
    attempts = 0

    async def fake_token(cls) -> str:
        return "tok"

    async def fake_parent_get_json(self, url, params=None, headers=None) -> dict:
        nonlocal attempts
        attempts += 1
        raise _TidalRateLimitError(0.01)

    async def fake_sleep(seconds: float) -> None:
        return None

    monkeypatch.setattr(TidalBase, "_ensure_token", classmethod(fake_token))
    monkeypatch.setattr(BaseScraper, "get_json", fake_parent_get_json)
    monkeypatch.setattr("salmon.sources.tidal.asyncio.sleep", fake_sleep)

    with pytest.raises(ScrapeError):
        asyncio.run(TidalBase().get_json("/albums/1"))

    assert attempts == _MAX_RATE_LIMIT_RETRIES
