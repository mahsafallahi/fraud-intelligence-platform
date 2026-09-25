import logging

import pandas as pd
import pytest
import requests

from src.ingestion.bronze_census_acs import (
    KEY_ENV_VAR,
    SOURCE_NAME,
    CensusAPIError,
    ingest,
    partition_path,
)
from tests.http_fakes import FakeResponse, FakeSession

FAKE_KEY = "fakekey0123456789abcdef"
# Shaped like the real response, including the -666666666 "not available" sentinel.
BODY = (
    '[["NAME","B01003_001E","B19013_001E","B01002_001E","state","zip code tabulation area"],\n'
    '["ZCTA5 00601","17113","14361","41.9","72","00601"],\n'
    '["ZCTA5 99999","0","-666666666","-666666666.0","02","99999"]]'
)


def ok_session():
    return FakeSession(FakeResponse(200, BODY))


def all_stored_text(bronze_dir) -> str:
    df = pd.read_parquet(partition_path(2019, bronze_dir))
    return " ".join(df.astype(str).to_numpy().ravel())


def test_stores_raw_body_exactly(tmp_path):
    assert ingest(2019, tmp_path, ok_session(), FAKE_KEY) == 2

    df = pd.read_parquet(partition_path(2019, tmp_path))
    assert len(df) == 1
    assert df["response_body"].iloc[0] == BODY  # sentinel kept as received


def test_sends_key_but_never_stores_it(tmp_path):
    session = ok_session()
    ingest(2019, tmp_path, session, FAKE_KEY)

    assert session.params[0]["key"] == FAKE_KEY
    row = pd.read_parquet(partition_path(2019, tmp_path)).iloc[0]
    assert "key=" not in row["_request_url"]
    assert "zip+code+tabulation+area" in row["_request_url"]
    assert FAKE_KEY not in all_stored_text(tmp_path)


def test_adds_metadata(tmp_path):
    ingest(2019, tmp_path, ok_session(), FAKE_KEY)

    row = pd.read_parquet(partition_path(2019, tmp_path)).iloc[0]
    assert row["_source"] == SOURCE_NAME
    assert row["_request_url"].startswith("https://api.census.gov/data/2019/acs/acs5?")
    assert row["_vintage"] == 2019
    assert row["_ingested_at"].tzinfo is not None


def test_reads_key_from_environment(tmp_path, monkeypatch):
    monkeypatch.setenv(KEY_ENV_VAR, FAKE_KEY)
    session = ok_session()
    ingest(2019, tmp_path, session)
    assert session.params[0]["key"] == FAKE_KEY


def test_missing_key_fails_before_any_request(tmp_path, monkeypatch):
    monkeypatch.delenv(KEY_ENV_VAR, raising=False)
    session = ok_session()
    with pytest.raises(CensusAPIError, match=KEY_ENV_VAR):
        ingest(2019, tmp_path, session)
    assert session.urls == []


@pytest.mark.parametrize(
    "error",
    [
        requests.HTTPError(f"400 Client Error for url: https://api.census.gov/data?key={FAKE_KEY}"),
        requests.ConnectionError(f"Max retries exceeded with url: /data/2019/acs/acs5?key={FAKE_KEY}"),
    ],
    ids=["http-error", "connection-error"],
)
def test_request_errors_are_redacted(tmp_path, error):
    with pytest.raises(CensusAPIError) as exc_info:
        ingest(2019, tmp_path, FakeSession(error=error), FAKE_KEY)

    assert FAKE_KEY not in str(exc_info.value)
    assert "***" in str(exc_info.value)
    # the original exception (with the key) is not chained into the traceback
    assert exc_info.value.__cause__ is None
    assert exc_info.value.__suppress_context__


def test_urllib3_log_messages_are_redacted(tmp_path, caplog):
    ingest(2019, tmp_path, ok_session(), FAKE_KEY)

    with caplog.at_level(logging.DEBUG):
        # what urllib3 logs on a retry / on every request at DEBUG level
        logging.getLogger("urllib3.connectionpool").warning(
            "Retrying (%r) after connection broken by '%r': %s",
            "Retry(total=2)", "timeout", f"/data/2019/acs/acs5?key={FAKE_KEY}",
        )
        logging.getLogger("urllib3.connectionpool").debug(
            '%s://%s:%s "%s %s %s" %s', "https", "api.census.gov", 443,
            "GET", f"/data/2019/acs/acs5?key={FAKE_KEY}", "HTTP/1.1", 200,
        )

    assert FAKE_KEY not in caplog.text
    assert caplog.text.count("key=***") == 2


@pytest.mark.parametrize(
    "response",
    [
        FakeResponse(200, "<html><title>Invalid Key</title></html>"),
        FakeResponse(200, '[["NAME","B01003_001E"]]'),  # header only
        FakeResponse(200, '[["NAME","state","zip code tabulation area"],["x","1","00601"]]'),  # missing variables
        FakeResponse(200, '{"error": "unknown variable"}'),
    ],
    ids=["html-page", "no-rows", "missing-columns", "not-a-list"],
)
def test_bad_response_fails_and_keeps_existing_partition(tmp_path, response):
    ingest(2019, tmp_path, ok_session(), FAKE_KEY)

    with pytest.raises(ValueError) as exc_info:
        ingest(2019, tmp_path, FakeSession(response), FAKE_KEY)
    assert FAKE_KEY not in str(exc_info.value)

    df = pd.read_parquet(partition_path(2019, tmp_path))
    assert df["response_body"].iloc[0] == BODY
