"""
Proxy sessions, and replacing them when YouTube burns one.

YouTube refuses its player API outright from cloud datacenter ranges, so on the
platform every player and caption request goes through a proxy. Individual
proxy addresses then get challenged unpredictably: the same video that fails on
one address succeeds on the next.

A single shared HTTP client is the wrong shape for that. Its connection pool
keeps a handful of connections alive, so every concurrent worker ends up
sharing the same few addresses and they get burned together. Instead each
worker owns a `Fetcher` — one client pinned to one named proxy session, which
Apify maps to a stable address — and when that address is challenged the worker
throws the session away and takes a fresh one. One bad address costs one
rotation instead of failing the run.
"""

from __future__ import annotations

import random
import string

import httpx

_TIMEOUT = httpx.Timeout(30.0, connect=15.0)


def _session_id() -> str:
    """A fresh Apify proxy session name (alphanumerics only)."""
    return "yt" + "".join(random.choices(string.ascii_lowercase + string.digits, k=12))


class Fetcher:
    """One HTTP client bound to one proxy address, replaceable on demand."""

    def __init__(self, configuration):
        self._configuration = configuration
        self._client: httpx.AsyncClient | None = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise RuntimeError("Fetcher used before open()")
        return self._client

    async def open(self) -> None:
        proxy = None
        if self._configuration is not None:
            try:
                proxy = await self._configuration.new_url(_session_id())
            except Exception:  # noqa: BLE001 - fall back to a direct connection
                proxy = None
        self._client = httpx.AsyncClient(
            proxy=proxy,
            timeout=_TIMEOUT,
            follow_redirects=True,
            # One address per worker: pooling more connections here would put
            # this worker back on the shared addresses rotation exists to avoid.
            limits=httpx.Limits(max_connections=2, max_keepalive_connections=1),
            headers={"Accept-Encoding": "gzip"},
        )

    async def rotate(self) -> None:
        """Discard the current address and take a new one."""
        await self.close()
        await self.open()

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None


class Pool:
    """The set of fetchers the workers draw from, one per concurrent slot."""

    def __init__(self, configuration, size: int):
        self._configuration = configuration
        self._size = max(1, size)
        self.fetchers: list[Fetcher] = []

    async def __aenter__(self) -> "Pool":
        for _ in range(self._size):
            fetcher = Fetcher(self._configuration)
            await fetcher.open()
            self.fetchers.append(fetcher)
        return self

    async def __aexit__(self, *exc) -> bool:
        for fetcher in self.fetchers:
            await fetcher.close()
        return False
