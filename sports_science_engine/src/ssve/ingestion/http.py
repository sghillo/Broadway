"""Polite HTTP client shared by all ingestion sources (retries, rate limiting, UA)."""
from __future__ import annotations

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

DEFAULT_USER_AGENT = "SSVE-Ingestion/0.1 (+research use; contact via repository)"


class PoliteSession:
    """requests.Session wrapper enforcing a minimum interval between calls.

    NCBI allows 3 req/s without an API key and 10 req/s with one, so the default
    interval (0.34 s) keeps us inside the anonymous limit.
    """

    def __init__(self, min_interval: float = 0.34, timeout: float = 30.0,
                 user_agent: str = DEFAULT_USER_AGENT, retries: int = 3) -> None:
        self.min_interval = min_interval
        self.timeout = timeout
        self._last_call = 0.0
        self.session = requests.Session()
        self.session.headers["User-Agent"] = user_agent
        retry = Retry(total=retries, backoff_factor=1.0,
                      status_forcelist=(429, 500, 502, 503, 504),
                      allowed_methods=("GET", "POST"))
        adapter = HTTPAdapter(max_retries=retry)
        self.session.mount("https://", adapter)
        self.session.mount("http://", adapter)

    def _throttle(self) -> None:
        wait = self.min_interval - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def get(self, url: str, **kwargs) -> requests.Response:
        self._throttle()
        kwargs.setdefault("timeout", self.timeout)
        response = self.session.get(url, **kwargs)
        response.raise_for_status()
        return response
