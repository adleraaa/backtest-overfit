"""Download and parse monthly tables from the Kenneth R. French Data Library.

Source: https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
The library does not state a redistribution license, so this project only commits the
download code; raw files are cached under data/raw/ (gitignored).
"""

from __future__ import annotations

import hashlib
import re
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"

DATASETS = {
    "industries49": "49_Industry_Portfolios_CSV.zip",
    "ff3": "F-F_Research_Data_Factors_CSV.zip",
    "ff5": "F-F_Research_Data_5_Factors_2x3_CSV.zip",
}

DEFAULT_CACHE = Path("data/raw")

# The library marks missing observations with these sentinels.
_MISSING = (-99.99, -999.0)
_MONTH_ROW = re.compile(r"^\d{6}$")


def download(name: str, cache_dir: Path = DEFAULT_CACHE, refresh: bool = False) -> Path:
    """Return the path of the cached zip for ``name``, downloading it if needed."""
    filename = DATASETS[name]
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / filename
    if refresh or not path.exists():
        request = urllib.request.Request(BASE_URL + filename, headers={"User-Agent": "btoverfit"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read()
        tmp = path.with_suffix(".part")
        tmp.write_bytes(payload)
        # The server answers maintenance or error pages with HTML; caching that as a .zip
        # would break every later run, so check before it replaces the cached file.
        if not zipfile.is_zipfile(tmp):
            tmp.unlink()
            raise ValueError(f"{BASE_URL + filename} did not return a zip file")
        tmp.replace(path)  # atomic-ish: never leave a truncated zip in the cache
    return path


def sha256(path: Path) -> str:
    """Hex digest of a cached file. The library revises history every month, so the
    digest recorded with the results says exactly which download produced them."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_zip_text(path: Path) -> str:
    """Each library zip holds exactly one CSV; return its text."""
    with zipfile.ZipFile(path) as zf:
        (member,) = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        return zf.read(member).decode("latin-1")


def data_vintage(text: str) -> str:
    """First header line, e.g. 'This file was created using the 202608 CRSP database.'"""
    return text.lstrip().splitlines()[0].strip()


def parse_monthly_table(text: str) -> pd.DataFrame:
    """Parse the first monthly table of a French library CSV into decimal returns.

    The files contain several tables back to back (monthly value-weighted, monthly
    equal-weighted, annual, ...). The first table is the monthly one we want: it starts at
    the first line beginning with a comma (the column header) and ends at the first row
    whose first field is not a YYYYMM date.
    """
    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith(","))
    columns = [c.strip() for c in lines[header_idx].split(",")[1:]]

    dates: list[str] = []
    values: list[list[float]] = []
    for line in lines[header_idx + 1 :]:
        fields = [f.strip() for f in line.split(",")]
        if not _MONTH_ROW.match(fields[0]):
            break
        dates.append(fields[0])
        values.append([float(v) for v in fields[1:]])

    data = np.array(values, dtype=float)
    data[np.isin(data, _MISSING)] = np.nan
    index = pd.PeriodIndex([pd.Period(f"{d[:4]}-{d[4:]}", freq="M") for d in dates], name="month")
    return pd.DataFrame(data / 100.0, index=index, columns=columns)


def load(
    name: str, cache_dir: Path = DEFAULT_CACHE, refresh: bool = False
) -> tuple[pd.DataFrame, str]:
    """Load one dataset as (monthly decimal returns, vintage string)."""
    text = read_zip_text(download(name, cache_dir, refresh=refresh))
    return parse_monthly_table(text), data_vintage(text)
