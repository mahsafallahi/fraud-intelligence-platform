"""Read access to the serving snapshot (data/serving/serving.duckdb), Phase 6.

The snapshot is built from Gold by scripts/build_serving_snapshot.py, mounted
read-only and never written by the API: scored transactions do NOT update it
(see docs/api_design.md for the consequence on card_txn_1h / card_txn_24h).
One read-only DuckDB connection per process; each request uses its own cursor,
which is DuckDB's way to share a connection between threads.
"""

from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import duckdb
import pandas as pd

from src.api.resolve import FRAME_SQL, SCHEMA_VERSION, frame_params

TABLES = ("card_history", "cards", "calendar", "state_holidays", "categories", "meta")


class SnapshotError(RuntimeError):
    """The snapshot is missing, incomplete or of another schema version."""


class UnknownCard(LookupError):
    pass


class DateOutsideCalendar(LookupError):
    pass


@dataclass(frozen=True)
class Calendar:
    first: date
    last: date


class SnapshotStore:
    def __init__(self, path, threads: int = 1, memory_limit: str = "512MB"):
        path = Path(path)
        if not path.is_file():
            raise SnapshotError(f"snapshot not found: {path}")
        self.path = path
        self._con = duckdb.connect(
            str(path), read_only=True, config={"threads": threads, "memory_limit": memory_limit})
        tables = {r[0] for r in self._con.execute(
            "select table_name from information_schema.tables").fetchall()}
        missing = sorted(set(TABLES) - tables)
        if missing:
            raise SnapshotError(f"snapshot is missing tables: {missing}")
        self.meta = dict(self._con.execute("select key, value from meta").fetchall())
        if self.meta.get("schema_version") != str(SCHEMA_VERSION):
            raise SnapshotError(f"snapshot schema_version {self.meta.get('schema_version')}, "
                                f"expected {SCHEMA_VERSION}")
        first, last = self._con.execute(
            "select min(calendar_date), max(calendar_date) from calendar").fetchone()
        self.calendar = Calendar(first, last)

    def ping(self) -> None:
        self._con.cursor().execute("select 1").fetchone()

    def has_card(self, card_key: str) -> bool:
        cur = self._con.cursor()
        return cur.execute("select 1 from cards where card_key = ?", [card_key]).fetchone() is not None

    def model_frame(self, card_key: str, trans_ts: datetime, amt: float, category: str,
                    merch_lat: float, merch_long: float) -> pd.DataFrame:
        """The card's earlier transactions plus the new one, in model-frame columns.

        The last column, is_new, marks the new transaction (exactly one row).
        """
        if not self.calendar.first <= trans_ts.date() <= self.calendar.last:
            raise DateOutsideCalendar(
                f"date {trans_ts.date()} is outside the calendar {self.calendar.first}..{self.calendar.last}")
        if not self.has_card(card_key):
            raise UnknownCard("unknown card")
        cur = self._con.cursor()
        frame = cur.execute(FRAME_SQL, frame_params(
            card_key, trans_ts, amt, category, merch_lat, merch_long)).df()
        n_new = int(frame["is_new"].sum())
        if n_new != 1:  # the lookups above make this impossible; never score a guess
            raise SnapshotError(f"expected one new row, got {n_new}")
        return frame

    def close(self) -> None:
        self._con.close()
