import pandas as pd
import pytest
import requests

from src.ingestion.bronze_holidays import (
    SOURCE_NAME,
    ingest_year,
    make_session,
    partition_path,
)

# Shaped like the real API response, including a null field and a list field.
BODY = (
    '[{"date":"2019-01-01","localName":"New Year\'s Day","name":"New Year\'s Day",'
    '"countryCode":"US","fixed":false,"global":true,"counties":null,"launchYear":null,'
    '"types":["Public","Bank"]},'
    '{"date":"2019-02-12","localName":"Lincoln\'s Birthday","name":"Lincoln\'s Birthday",'
    '"countryCode":"US","fixed":false,"global":false,"counties":["US-CA","US-NY"],'
    '"launchYear":null,"types":["Observance"]}]'
)


class FakeResponse:
    def __init__(self, status_code: int, text: str):
        self.status_code = status_code
        self.text = text

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code} error")


class FakeSession:
    def __init__(self, response: FakeResponse):
        self.response = response
        self.urls = []

    def get(self, url, timeout):
        self.urls.append(url)
        return self.response


def test_stores_raw_body_exactly(tmp_path):
    session = FakeSession(FakeResponse(200, BODY))
    assert ingest_year(2019, "US", tmp_path, session) == 2

    df = pd.read_parquet(partition_path("US", 2019, tmp_path))
    assert len(df) == 1
    assert df["response_body"].iloc[0] == BODY
    assert session.urls == ["https://date.nager.at/api/v3/PublicHolidays/2019/US"]


def test_adds_metadata(tmp_path):
    ingest_year(2019, "US", tmp_path, FakeSession(FakeResponse(200, BODY)))

    row = pd.read_parquet(partition_path("US", 2019, tmp_path)).iloc[0]
    assert row["_source"] == SOURCE_NAME
    assert row["_request_url"].endswith("/2019/US")
    assert row["_country"] == "US"
    assert row["_year"] == 2019
    assert row["_ingested_at"].tzinfo is not None


def test_rerun_overwrites(tmp_path):
    for _ in range(2):
        ingest_year(2019, "US", tmp_path, FakeSession(FakeResponse(200, BODY)))

    part_dir = partition_path("US", 2019, tmp_path).parent
    assert [p.name for p in part_dir.iterdir()] == ["part-0.parquet"]
    assert len(pd.read_parquet(part_dir)) == 1


@pytest.mark.parametrize(
    "response, error",
    [
        (FakeResponse(404, '{"title":"Unknown country code"}'), requests.HTTPError),
        (FakeResponse(200, "<html>maintenance</html>"), ValueError),
        (FakeResponse(200, "[]"), ValueError),
        (FakeResponse(200, '{"unexpected": "object"}'), ValueError),
    ],
    ids=["http-404", "not-json", "empty-list", "not-a-list"],
)
def test_bad_response_fails_and_keeps_existing_partition(tmp_path, response, error):
    ingest_year(2019, "US", tmp_path, FakeSession(FakeResponse(200, BODY)))

    with pytest.raises(error):
        ingest_year(2019, "US", tmp_path, FakeSession(response))

    # the good data from the first run is still there
    df = pd.read_parquet(partition_path("US", 2019, tmp_path))
    assert df["response_body"].iloc[0] == BODY


def test_session_retries_only_transient_errors():
    retry = make_session().get_adapter("https://date.nager.at").max_retries
    assert retry.total == 3
    assert {429, 500, 503}.issubset(retry.status_forcelist)
    assert 404 not in retry.status_forcelist
