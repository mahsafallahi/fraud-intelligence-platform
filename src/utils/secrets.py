"""Helpers to keep secrets (e.g. API keys sent as URL parameters) out of logs and errors."""

import logging

MASK = "***"


def redact(text: str, secret: str | None) -> str:
    return text.replace(secret, MASK) if secret else text


class RedactSecretFilter(logging.Filter):
    """Replaces the secret in a log record's message before any handler sees it."""

    def __init__(self, secret: str):
        super().__init__()
        self.secret = secret

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        if self.secret in message:
            record.msg, record.args = redact(message, self.secret), None
        return True


def mask_secret_in_logs(secret: str, logger_names: list[str]) -> None:
    """Attach a redacting filter to each named logger (idempotent).

    A filter on a logger applies to records created by that logger, before they
    propagate to any handler, so this works whatever logging setup the caller
    has (CLI, pytest, Airflow).
    """
    for name in logger_names:
        logger = logging.getLogger(name)
        if not any(isinstance(f, RedactSecretFilter) and f.secret == secret for f in logger.filters):
            logger.addFilter(RedactSecretFilter(secret))
