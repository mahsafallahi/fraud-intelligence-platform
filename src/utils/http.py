import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

TIMEOUT_SECONDS = 30


def make_session() -> requests.Session:
    """HTTP session that retries transient failures only.

    Network errors, 429 and 5xx are retried with exponential backoff; anything
    else (e.g. 404) fails immediately, because retrying will not fix it.
    """
    retry = Retry(
        total=3,
        backoff_factor=1,  # waits 1s, 2s, 4s between attempts
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session
