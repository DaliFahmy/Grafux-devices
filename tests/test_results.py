"""
test_results.py
Unit tests for device.results.ResultStore — the TTL result cache.
"""

import time

import pytest

from device.results import ResultStore


def test_put_peek_pop_consume_once():
    s = ResultStore(ttl_s=60)
    s.put("c1", {"v": 1})
    assert s.peek("c1") == {"v": 1}     # peek does not consume
    assert s.pop("c1") == {"v": 1}      # pop consumes
    assert s.pop("c1") is None          # gone after consume


def test_missing_key_returns_none():
    s = ResultStore(ttl_s=60)
    assert s.pop("nope") is None
    assert s.peek("nope") is None


def test_ttl_expiry_on_read():
    s = ResultStore(ttl_s=1)
    s.put("old", {"v": 3})
    time.sleep(1.1)
    assert s.pop("old") is None         # expired entries are not returned
    s.put("latest:d", {"v": 4})
    time.sleep(1.1)
    assert s.peek("latest:d") is None   # latest:* keys age out too (bounded memory)


async def test_sweeper_lifecycle_is_idempotent():
    s = ResultStore(ttl_s=1)
    s.start_sweeper()
    s.start_sweeper()                   # second call is a no-op, not a second task
    assert s._sweeper is not None
    await s.stop_sweeper()
    assert s._sweeper is None
    await s.stop_sweeper()              # safe to stop twice
