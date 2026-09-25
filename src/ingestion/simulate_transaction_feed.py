"""Simulate an upstream system that delivers one transactions file per day.

The Kaggle dataset is a static dump (fraudTrain.csv + fraudTest.csv). This
script replays it as a daily feed: it splits both files into
landing/transactions/YYYY-MM-DD.csv, one file per calendar day of
trans_date_trans_time.

It is NOT part of the pipeline; it stands in for the outside world. The Bronze
ingestion only ever reads the landing zone.

Days inside the range that have no rows get a header-only file, like a real
feed confirming "nothing today".

Lines are copied byte for byte (no CSV parsing), so each landing file is exactly
what the source contained for that day, and memory use stays constant.

Usage:
    python -m src.ingestion.simulate_transaction_feed
"""

import argparse
import logging
import shutil
from datetime import date, timedelta
from pathlib import Path

from src.utils.paths import LANDING_DIR, RAW_DIR

log = logging.getLogger(__name__)

SOURCE_FILES = ("fraudTrain.csv", "fraudTest.csv")


def _day_of(line: bytes) -> str:
    # Row layout: <index>,<trans_date_trans_time>,... ; neither of the first two
    # fields contains commas, so a split on the first two commas is safe.
    return line.split(b",", 2)[1][:10].decode("ascii")


def split_into_daily_files(source_paths: list[Path], out_dir: Path) -> dict[str, int]:
    """Split source CSVs into one CSV per day. Returns rows written per day.

    The output directory is rebuilt from scratch, so re-running is idempotent.
    Input does not need to be sorted: a day's file is reopened in append mode
    whenever that day shows up again.
    """
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    rows_per_day: dict[str, int] = {}
    header: bytes | None = None

    for src in source_paths:
        with src.open("rb") as f:
            src_header = f.readline()
            if header is None:
                header = src_header
            elif src_header != header:
                raise ValueError(f"{src.name} has a different header than {source_paths[0].name}")

            current_day, out = None, None
            for line in f:
                day = _day_of(line)
                if day != current_day:
                    if out:
                        out.close()
                    is_new = day not in rows_per_day
                    out = (out_dir / f"{day}.csv").open("ab")
                    if is_new:
                        out.write(header)
                        rows_per_day[day] = 0
                    current_day = day
                out.write(line)
                rows_per_day[day] += 1
            if out:
                out.close()

        log.info("split %s", src.name)

    # A real feed delivers a file every day, even when it has no rows (e.g. the
    # source has no data for 2020-02-29). Write header-only files for such days
    # so that a *missing* file keeps meaning "delivery failed".
    if rows_per_day:
        first, last = date.fromisoformat(min(rows_per_day)), date.fromisoformat(max(rows_per_day))
        for offset in range((last - first).days + 1):
            day = (first + timedelta(days=offset)).isoformat()
            if day not in rows_per_day:
                (out_dir / f"{day}.csv").write_bytes(header)
                rows_per_day[day] = 0
                log.warning("no rows for %s, wrote an empty file", day)

    return dict(sorted(rows_per_day.items()))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=LANDING_DIR / "transactions")
    args = parser.parse_args()

    rows = split_into_daily_files([args.raw_dir / name for name in SOURCE_FILES], args.out_dir)
    log.info(
        "wrote %d daily files (%s -> %s), %d rows total",
        len(rows), min(rows), max(rows), sum(rows.values()),
    )


if __name__ == "__main__":
    main()
