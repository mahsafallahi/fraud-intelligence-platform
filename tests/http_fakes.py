"""Minimal stand-ins for requests.Session / Response, so tests never hit the network."""

import requests


class FakeResponse:
    def __init__(self, status_code: int, body: str | bytes):
        self.status_code = status_code
        self.content = body.encode("utf-8") if isinstance(body, str) else body
        self.text = self.content.decode("utf-8", errors="replace")

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.urls = []

    def get(self, url, timeout, **kwargs):
        self.urls.append(url)
        return self.response
