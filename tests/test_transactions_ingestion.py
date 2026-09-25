from datetime import date

import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.ingestion.bronze_transactions import SOURCE_NAME, ingest_day, partition_path
from src.ingestion.simulate_transaction_feed import split_into_daily_files

HEADER = b",trans_date_trans_time,cc_num,merchant,amt,is_fraud\r\n"
TRAIN_ROWS = [
    b'0,2019-01-01 00:00:18,2703186189652095,"fraud_Rippin, Kub and Mann",4.97,0\r\n',
    b"1,2019-01-01 23:59:59,630423337322,fraud_Lind-Buckridge,,1\r\n",  # empty amt
    b"2,2019-01-02 00:01:00,38859492057661,fraud_Keeling-Crist,41.96,0\r\n",
    b"3,2019-01-01 12:00:00,375534208663984,fraud_Keeling-Crist,7.00,0\r\n",  # out of order
]
TEST_ROWS = [
    b"0,2019-01-02 10:00:00,630423337322,fraud_Lind-Buckridge,100.00,0\r\n",
]


@pytest.fixture
def raw_files(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "train.csv").write_bytes(HEADER + b"".join(TRAIN_ROWS))
    (raw / "test.csv").write_bytes(HEADER + b"".join(TEST_ROWS))
    return [raw / "train.csv", raw / "test.csv"]


@pytest.fixture
def landing_dir(tmp_path, raw_files):
    out = tmp_path / "landing" / "transactions"
    split_into_daily_files(raw_files, out)
    return out


# --- simulator ---------------------------------------------------------------

def test_split_creates_one_file_per_day_with_exact_bytes(landing_dir):
    assert sorted(p.name for p in landing_dir.iterdir()) == ["2019-01-01.csv", "2019-01-02.csv"]
    assert (landing_dir / "2019-01-01.csv").read_bytes() == (
        HEADER + TRAIN_ROWS[0] + TRAIN_ROWS[1] + TRAIN_ROWS[3]
    )
    # a day spanning both source files is merged into one file, header written once
    assert (landing_dir / "2019-01-02.csv").read_bytes() == HEADER + TRAIN_ROWS[2] + TEST_ROWS[0]


def test_split_is_idempotent(raw_files, landing_dir):
    rows = split_into_daily_files(raw_files, landing_dir)
    assert rows == {"2019-01-01": 3, "2019-01-02": 2}
    assert (landing_dir / "2019-01-01.csv").read_bytes().count(HEADER) == 1


def test_split_writes_empty_file_for_day_without_rows(tmp_path, raw_files):
    raw_files[1].write_bytes(HEADER + b"0,2019-01-04 09:00:00,1,m,1.00,0\r\n")
    out = tmp_path / "out"
    rows = split_into_daily_files(raw_files, out)

    assert rows["2019-01-03"] == 0
    assert (out / "2019-01-03.csv").read_bytes() == HEADER


def test_split_rejects_mismatched_headers(tmp_path, raw_files):
    raw_files[1].write_bytes(b",other,header\r\n" + TEST_ROWS[0])
    with pytest.raises(ValueError, match="different header"):
        split_into_daily_files(raw_files, tmp_path / "out")


# --- bronze ------------------------------------------------------------------

def test_bronze_keeps_values_as_received(tmp_path, landing_dir):
    bronze = tmp_path / "bronze"
    assert ingest_day(date(2019, 1, 1), landing_dir, bronze) == 3

    df = pd.read_parquet(partition_path(date(2019, 1, 1), bronze))
    business_cols = [c for c in df.columns if not c.startswith("_")]
    assert all(pd.api.types.is_string_dtype(df[c]) for c in business_cols)
    assert df["cc_num"].tolist()[0] == "2703186189652095"  # not turned into a number
    assert df["amt"].tolist() == ["4.97", "", "7.00"]  # empty stays empty, "7.00" keeps its zeros
    assert df["merchant"].tolist()[0] == "fraud_Rippin, Kub and Mann"


def test_bronze_adds_metadata(tmp_path, landing_dir):
    bronze = tmp_path / "bronze"
    ingest_day(date(2019, 1, 2), landing_dir, bronze)

    df = pd.read_parquet(partition_path(date(2019, 1, 2), bronze))
    assert (df["_source"] == SOURCE_NAME).all()
    assert (df["_source_file"].str.endswith("2019-01-02.csv")).all()
    assert (df["_batch_date"] == date(2019, 1, 2)).all()
    assert df["_ingested_at"].dt.tz is not None  # stored in UTC


def test_bronze_rerun_overwrites_instead_of_appending(tmp_path, landing_dir):
    bronze = tmp_path / "bronze"
    ingest_day(date(2019, 1, 1), landing_dir, bronze)
    ingest_day(date(2019, 1, 1), landing_dir, bronze)

    part_dir = partition_path(date(2019, 1, 1), bronze).parent
    assert [p.name for p in part_dir.iterdir()] == ["part-0.parquet"]
    assert len(pd.read_parquet(part_dir)) == 3


def test_bronze_empty_file_gives_empty_partition_with_same_schema(tmp_path, landing_dir):
    (landing_dir / "2019-01-03.csv").write_bytes(HEADER)
    bronze = tmp_path / "bronze"
    assert ingest_day(date(2019, 1, 3), landing_dir, bronze) == 0
    ingest_day(date(2019, 1, 1), landing_dir, bronze)

    empty = pq.read_schema(partition_path(date(2019, 1, 3), bronze))
    normal = pq.read_schema(partition_path(date(2019, 1, 1), bronze))
    assert empty.remove_metadata() == normal.remove_metadata()


def test_bronze_missing_landing_file_fails(tmp_path, landing_dir):
    with pytest.raises(FileNotFoundError, match="2019-01-03"):
        ingest_day(date(2019, 1, 3), landing_dir, tmp_path / "bronze")
