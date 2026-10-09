"""Tests for the Silver writer (src/pipeline/silver_jobs.py).

Bronze inputs are built with the helpers from the Silver transform tests, saved
as Bronze Parquet files in a temp folder, then run through the writer.
"""

import datetime
import logging
from pathlib import Path

import pandas as pd
import pytest

from src.ingestion.bronze_mcc import PINNED_COMMIT
from src.ingestion.bronze_mcc import partition_path as bronze_mcc_partition_path
from src.pipeline.silver_jobs import (
    PART_FILE,
    census_acs_partition,
    census_zcta_partition,
    fail_fast_on_validation_error,
    holidays_partition,
    mcc_partition,
    pinned_mcc_version,
    run_census_acs,
    run_census_zcta,
    run_holidays,
    run_mcc,
    run_transactions_day,
    transactions_partition,
    version_from_folder,
)
from src.silver.census_acs import transform_census_acs
from src.silver.census_zcta import transform_census_zcta
from src.silver.holidays import transform_holidays
from src.silver.mcc import SILVER_SCHEMA as MCC_SILVER_SCHEMA
from src.silver.mcc import transform_mcc
from src.silver.transactions import (
    SILVER_SCHEMA,
    SilverValidationError,
    transform_transactions,
)
from tests import test_silver_census_acs as acs
from tests import test_silver_census_zcta as zcta
from tests import test_silver_holidays as hol
from tests import test_silver_mcc as mccs
from tests.test_silver_transactions import KEY, make_bronze, make_row

DAY = datetime.date(2019, 1, 1)


def save_bronze(df: pd.DataFrame, bronze_dir, partition: str) -> None:
    path = bronze_dir / partition / PART_FILE
    path.parent.mkdir(parents=True)
    df.to_parquet(path, index=False)


def read_silver(silver_dir, partition: str) -> pd.DataFrame:
    return pd.read_parquet(silver_dir / partition / PART_FILE)


@pytest.fixture
def dirs(tmp_path):
    return tmp_path / "bronze", tmp_path / "silver"


def bronze_day():
    return make_bronze([
        make_row(),
        make_row(trans_num="t0002", zip="7640", trans_date_trans_time="2019-01-01 10:00:00"),
    ])


# --- 1. Transactions ---------------------------------------------------------------

def test_transactions_written_to_its_partition_with_exact_dtypes(dirs):
    bronze_dir, silver_dir = dirs
    save_bronze(bronze_day(), bronze_dir, transactions_partition(DAY))

    assert run_transactions_day(DAY, KEY, bronze_dir, silver_dir) == 2

    expected = transform_transactions(bronze_day(), KEY)
    written = read_silver(silver_dir, "transactions/batch_date=2019-01-01")
    assert written.dtypes.to_dict() == expected.dtypes.to_dict()
    assert written["_batch_date"].dtype == "date32[pyarrow]"
    pd.testing.assert_frame_equal(written, expected)


def test_empty_day_writes_zero_rows_with_full_schema(dirs):
    bronze_dir, silver_dir = dirs
    save_bronze(make_bronze([]), bronze_dir, transactions_partition(DAY))

    assert run_transactions_day(DAY, KEY, bronze_dir, silver_dir) == 0

    written = read_silver(silver_dir, transactions_partition(DAY))
    assert len(written) == 0
    assert list(written.columns) == list(SILVER_SCHEMA)
    assert written.dtypes.to_dict() == transform_transactions(bronze_day(), KEY).dtypes.to_dict()


def test_rerun_overwrites_instead_of_appending(dirs):
    bronze_dir, silver_dir = dirs
    save_bronze(bronze_day(), bronze_dir, transactions_partition(DAY))
    for _ in range(2):
        run_transactions_day(DAY, KEY, bronze_dir, silver_dir)

    part_dir = silver_dir / transactions_partition(DAY)
    assert [p.name for p in part_dir.iterdir()] == [PART_FILE]
    assert len(read_silver(silver_dir, transactions_partition(DAY))) == 2


def test_validation_error_keeps_existing_silver_partition(dirs):
    bronze_dir, silver_dir = dirs
    save_bronze(bronze_day(), bronze_dir, transactions_partition(DAY))
    run_transactions_day(DAY, KEY, bronze_dir, silver_dir)
    before = read_silver(silver_dir, transactions_partition(DAY))

    bad = make_bronze([make_row(amt="-1")])
    (bronze_dir / transactions_partition(DAY) / PART_FILE).unlink()
    bad.to_parquet(bronze_dir / transactions_partition(DAY) / PART_FILE, index=False)

    with pytest.raises(SilverValidationError):
        run_transactions_day(DAY, KEY, bronze_dir, silver_dir)
    pd.testing.assert_frame_equal(read_silver(silver_dir, transactions_partition(DAY)), before)


def test_missing_bronze_partition_fails(dirs):
    bronze_dir, silver_dir = dirs
    with pytest.raises(FileNotFoundError, match="2019-01-01"):
        run_transactions_day(DAY, KEY, bronze_dir, silver_dir)
    assert not silver_dir.exists()


def test_key_never_appears_in_output_or_logs(dirs, caplog):
    bronze_dir, silver_dir = dirs
    save_bronze(bronze_day(), bronze_dir, transactions_partition(DAY))

    with caplog.at_level(logging.DEBUG):
        run_transactions_day(DAY, KEY, bronze_dir, silver_dir)

    written = read_silver(silver_dir, transactions_partition(DAY))
    stored = " ".join(written.astype(str).to_numpy().ravel())
    for form in (KEY.hex(), str(KEY)):
        assert form not in stored
        assert form not in caplog.text


# --- 2. Reference sources -------------------------------------------------------------

def test_census_acs_partition_keeps_nullable_int64(dirs):
    bronze_dir, silver_dir = dirs
    rows = [acs.acs_row(), acs.acs_row(zip_code="99999", income="-666666666", age="-666666666")]
    bronze = acs.make_bronze(rows)
    save_bronze(bronze, bronze_dir, census_acs_partition(2019))

    assert run_census_acs(2019, bronze_dir, silver_dir) == 2

    written = read_silver(silver_dir, "census_acs/vintage=2019")
    expected = transform_census_acs(bronze)
    assert written["median_household_income"].dtype == "Int64"
    assert written["median_household_income"].isna().sum() == 1
    pd.testing.assert_frame_equal(written, expected)


def test_census_zcta_passes_vintage_and_keeps_dtypes(dirs):
    bronze_dir, silver_dir = dirs
    bronze = zcta.make_bronze()
    save_bronze(bronze, bronze_dir, census_zcta_partition(2019))

    assert run_census_zcta(2019, bronze_dir, silver_dir) == len(bronze)

    written = read_silver(silver_dir, "census_zcta/vintage=2019")
    assert (written["_vintage"] == 2019).all()
    pd.testing.assert_frame_equal(written, transform_census_zcta(bronze, 2019))


def test_holidays_partition_keeps_date32_and_bool(dirs):
    bronze_dir, silver_dir = dirs
    bronze = hol.make_bronze()
    save_bronze(bronze, bronze_dir, holidays_partition("US", 2019))

    rows = run_holidays("US", 2019, bronze_dir, silver_dir)

    written = read_silver(silver_dir, "holidays/country=US/year=2019")
    assert rows == len(written) > 0
    assert written["holiday_date"].dtype == "date32[pyarrow]"
    assert written["is_global"].dtype == "bool"
    pd.testing.assert_frame_equal(written, transform_holidays(bronze))


def test_mcc_written_to_its_version_partition_with_exact_dtypes(dirs):
    bronze_dir, silver_dir = dirs
    bronze = mccs.make_bronze()
    save_bronze(bronze, bronze_dir, mcc_partition("9675cfab77d6"))

    assert run_mcc("9675cfab77d6", bronze_dir, silver_dir) == 3

    written = read_silver(silver_dir, "mcc/version=9675cfab77d6")
    assert {c: str(t) for c, t in written.dtypes.items()} == MCC_SILVER_SCHEMA
    assert written["mcc"].tolist() == ["0742", "5411", "5999"]  # leading zero kept
    assert (written["_version"] == "9675cfab77d6").all()
    assert "version" not in written.columns
    pd.testing.assert_frame_equal(written, transform_mcc(bronze, "9675cfab77d6"))


def test_mcc_rerun_overwrites_and_bad_batch_keeps_existing(dirs):
    bronze_dir, silver_dir = dirs
    save_bronze(mccs.make_bronze(), bronze_dir, mcc_partition("9675cfab77d6"))
    for _ in range(2):
        run_mcc("9675cfab77d6", bronze_dir, silver_dir)
    part_dir = silver_dir / mcc_partition("9675cfab77d6")
    assert [p.name for p in part_dir.iterdir()] == [PART_FILE]
    before = read_silver(silver_dir, mcc_partition("9675cfab77d6"))

    bad = mccs.make_bronze([{"mcc": "54A1"}])
    bad.to_parquet(bronze_dir / mcc_partition("9675cfab77d6") / PART_FILE, index=False)
    with pytest.raises(SilverValidationError, match="is not 4 digits"):
        run_mcc("9675cfab77d6", bronze_dir, silver_dir)
    pd.testing.assert_frame_equal(read_silver(silver_dir, mcc_partition("9675cfab77d6")), before)


def test_mcc_missing_bronze_partition_fails(dirs):
    bronze_dir, silver_dir = dirs
    with pytest.raises(FileNotFoundError, match="version=9675cfab77d6"):
        run_mcc("9675cfab77d6", bronze_dir, silver_dir)


def test_version_comes_from_the_folder_name():
    assert version_from_folder("version=9675cfab77d6") == "9675cfab77d6"
    for bad in ["vintage=2019", "version=", "9675cfab77d6", "versions=9675cfab77d6"]:
        with pytest.raises(ValueError, match="version=<hash>"):
            version_from_folder(bad)


def test_pinned_version_is_the_folder_bronze_writes():
    folder = bronze_mcc_partition_path(PINNED_COMMIT, Path("x")).parent.name
    assert folder == f"version={pinned_mcc_version()}"
    assert PINNED_COMMIT.startswith(pinned_mcc_version())


def test_reading_the_file_adds_no_partition_column(dirs):
    # Reading the partition folder could add `vintage` as a column, which the
    # ACS column contract rejects; the writer reads the file.
    bronze_dir, silver_dir = dirs
    save_bronze(acs.make_bronze([acs.acs_row()]), bronze_dir, census_acs_partition(2019))
    run_census_acs(2019, bronze_dir, silver_dir)
    assert "vintage" not in read_silver(silver_dir, census_acs_partition(2019)).columns


# --- 3. Fail fast on data errors --------------------------------------------------

class FakeFailException(Exception):
    """Stands in for airflow.sdk.exceptions.AirflowFailException."""


def test_validation_error_becomes_fail_exception(dirs):
    bronze_dir, silver_dir = dirs
    save_bronze(make_bronze([make_row(amt="-1")]), bronze_dir, transactions_partition(DAY))
    run = fail_fast_on_validation_error(FakeFailException)(run_transactions_day)

    with pytest.raises(FakeFailException, match="Silver validation failed: amt") as info:
        run(DAY, KEY, bronze_dir, silver_dir)
    assert isinstance(info.value.__cause__, SilverValidationError)


def test_other_errors_are_not_converted_so_retries_apply(dirs):
    bronze_dir, silver_dir = dirs
    run = fail_fast_on_validation_error(FakeFailException)(run_transactions_day)

    with pytest.raises(FileNotFoundError):
        run(DAY, KEY, bronze_dir, silver_dir)


def test_wrapper_returns_the_result_and_keeps_the_name():
    run = fail_fast_on_validation_error(FakeFailException)(run_transactions_day)
    assert run.__name__ == "run_transactions_day"
    assert fail_fast_on_validation_error(FakeFailException)(lambda: 7)() == 7
