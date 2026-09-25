import hashlib

import pandas as pd
import pytest
import requests

from src.ingestion.bronze_mcc import SOURCE_NAME, ingest, partition_path
from tests.http_fakes import FakeResponse, FakeSession

COMMIT = "9675cfab77d6160397d2b84c0ebc934439841929"
CSV = (
    b"mcc,edited_description,combined_description,usda_description,irs_description,irs_reportable\n"
    b"0742,Veterinary Services,Veterinary Services,Veterinary Services,Veterinary Services,Yes\n"
    b'0780,"Horticultural Services, Landscaping Services",x,y,z,Yes\n'
    b"5411,Grocery Stores,Grocery Stores,,Grocery Stores,\n"
)


def test_keeps_values_as_received(tmp_path):
    assert ingest(COMMIT, tmp_path, FakeSession(FakeResponse(200, CSV))) == 3

    df = pd.read_parquet(partition_path(COMMIT, tmp_path))
    assert df["mcc"].tolist() == ["0742", "0780", "5411"]  # leading zeros kept
    assert df["edited_description"].iloc[1] == "Horticultural Services, Landscaping Services"
    assert df["usda_description"].iloc[2] == ""  # empty stays empty, not NaN
    business_cols = [c for c in df.columns if not c.startswith("_")]
    assert all(pd.api.types.is_string_dtype(df[c]) for c in business_cols)


def test_adds_metadata_and_checksum(tmp_path):
    session = FakeSession(FakeResponse(200, CSV))
    ingest(COMMIT, tmp_path, session)

    df = pd.read_parquet(partition_path(COMMIT, tmp_path))
    assert session.urls == [f"https://raw.githubusercontent.com/greggles/mcc-codes/{COMMIT}/mcc_codes.csv"]
    assert (df["_source"] == SOURCE_NAME).all()
    assert (df["_source_url"] == session.urls[0]).all()
    assert (df["_content_sha256"] == hashlib.sha256(CSV).hexdigest()).all()
    assert df["_ingested_at"].dt.tz is not None


def test_partition_is_keyed_by_version_and_rerun_overwrites(tmp_path):
    for _ in range(2):
        ingest(COMMIT, tmp_path, FakeSession(FakeResponse(200, CSV)))

    part_dir = partition_path(COMMIT, tmp_path).parent
    assert part_dir.name == "version=9675cfab77d6"
    assert [p.name for p in part_dir.iterdir()] == ["part-0.parquet"]
    assert len(pd.read_parquet(part_dir)) == 3


@pytest.mark.parametrize(
    "response, error",
    [
        (FakeResponse(404, "404: Not Found"), requests.HTTPError),
        (FakeResponse(200, b"code,description\n0742,Vet\n"), ValueError),  # no mcc column
        (FakeResponse(200, CSV.splitlines(keepends=True)[0]), ValueError),  # header only
    ],
    ids=["http-404", "missing-mcc-column", "no-rows"],
)
def test_bad_download_fails_and_keeps_existing_partition(tmp_path, response, error):
    ingest(COMMIT, tmp_path, FakeSession(FakeResponse(200, CSV)))

    with pytest.raises(error):
        ingest(COMMIT, tmp_path, FakeSession(response))

    assert len(pd.read_parquet(partition_path(COMMIT, tmp_path))) == 3
