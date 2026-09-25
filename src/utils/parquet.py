import os
from pathlib import Path

import pandas as pd


def write_parquet_atomic(df: pd.DataFrame, dest: Path) -> None:
    """Write df to dest, replacing any existing file.

    Writes to a temp file first and then renames it over the target, so a crash
    mid-write never leaves a half-written file behind.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".parquet.tmp")
    df.to_parquet(tmp, index=False)
    os.replace(tmp, dest)
