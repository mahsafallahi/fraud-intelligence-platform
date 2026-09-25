import hashlib
import io
import zipfile

import pandas as pd
import pytest
import requests

from src.ingestion.bronze_census_zcta import SOURCE_NAME, ingest, partition_path
from tests.http_fakes import FakeResponse, FakeSession

# Shaped like the real file: tab-separated, with trailing spaces on every line.
TXT = (
    b"GEOID\tALAND\tAWATER\tALAND_SQMI\tAWATER_SQMI\tINTPTLAT\tINTPTLONG   \n"
    b"00601\t166659747\t799292\t64.348\t0.309\t18.180555\t-66.749961   \n"
    b"28654\t100\t0\t1.0\t0.0\t36.0\t-81.1   \n"
)


def make_zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


ZIP = make_zip({"2019_Gaz_zcta_national.txt": TXT})


def test_keeps_values_as_received(tmp_path):
    assert ingest(2019, tmp_path, FakeSession(FakeResponse(200, ZIP))) == 2

    df = pd.read_parquet(partition_path(2019, tmp_path))
    assert df["GEOID"].tolist() == ["00601", "28654"]  # leading zeros kept
    assert "INTPTLONG   " in df.columns  # trailing whitespace kept, trimmed in Silver
    assert df["INTPTLONG   "].iloc[0] == "-66.749961   "
    business_cols = [c for c in df.columns if not c.startswith("_")]
    assert all(pd.api.types.is_string_dtype(df[c]) for c in business_cols)


def test_adds_metadata_and_checksum(tmp_path):
    session = FakeSession(FakeResponse(200, ZIP))
    ingest(2019, tmp_path, session)

    df = pd.read_parquet(partition_path(2019, tmp_path))
    assert session.urls[0].endswith("/2019_Gazetteer/2019_Gaz_zcta_national.zip")
    assert (df["_source"] == SOURCE_NAME).all()
    assert (df["_source_url"] == session.urls[0]).all()
    assert (df["_source_file"] == "2019_Gaz_zcta_national.txt").all()
    assert (df["_content_sha256"] == hashlib.sha256(ZIP).hexdigest()).all()
    assert df["_ingested_at"].dt.tz is not None


def test_rerun_overwrites(tmp_path):
    for _ in range(2):
        ingest(2019, tmp_path, FakeSession(FakeResponse(200, ZIP)))

    part_dir = partition_path(2019, tmp_path).parent
    assert part_dir.name == "vintage=2019"
    assert [p.name for p in part_dir.iterdir()] == ["part-0.parquet"]
    assert len(pd.read_parquet(part_dir)) == 2


@pytest.mark.parametrize(
    "response, error",
    [
        (FakeResponse(404, "Not Found"), requests.HTTPError),
        (FakeResponse(200, "<html>Missing Key</html>"), ValueError),  # 200 but not a zip
        (FakeResponse(200, make_zip({"a.txt": TXT, "b.txt": TXT})), ValueError),
        (FakeResponse(200, make_zip({"x.txt": b"ZIP\tALAND\n00601\t1\n"})), ValueError),  # no GEOID
        (FakeResponse(200, make_zip({"x.txt": TXT.splitlines(keepends=True)[0]})), ValueError),
    ],
    ids=["http-404", "not-a-zip", "two-txt-files", "missing-geoid", "no-rows"],
)
def test_bad_download_fails_and_keeps_existing_partition(tmp_path, response, error):
    ingest(2019, tmp_path, FakeSession(FakeResponse(200, ZIP)))

    with pytest.raises(error):
        ingest(2019, tmp_path, FakeSession(response))

    assert len(pd.read_parquet(partition_path(2019, tmp_path))) == 2
