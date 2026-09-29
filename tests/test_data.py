import io
import zipfile

import numpy as np
import pandas as pd

from btoverfit.data import data_vintage, parse_monthly_table, read_zip_text

# Same layout as the real library files (header notes, a monthly table, then more tables),
# but with made-up numbers so no library data is redistributed in the repo.
FIXTURE = """This file was created using the 202608 CRSP database.
It contains value- and equal-weighted returns for 3 industry portfolios.

Missing data are indicated by -99.99 or -999.


  Average Value Weighted Returns -- Monthly
,Agric,Food ,Soda
192607,   2.00,  -1.00, -99.99
192608,   0.50,   3.25, -999
192609,  -4.00,   0.00,   1.50

  Average Equal Weighted Returns -- Monthly
,Agric,Food ,Soda
192607,   9.00,   9.00,   9.00
"""


def test_parse_monthly_table_reads_only_first_table():
    df = parse_monthly_table(FIXTURE)
    assert list(df.columns) == ["Agric", "Food", "Soda"]  # trailing spaces stripped
    assert len(df) == 3  # stops before the equal-weighted table
    assert df.index[0] == pd.Period("1926-07", freq="M")
    assert df.index[-1] == pd.Period("1926-09", freq="M")


def test_parse_monthly_table_converts_percent_and_missing():
    df = parse_monthly_table(FIXTURE)
    assert df.loc["1926-07", "Agric"] == 0.02
    assert df.loc["1926-08", "Food"] == 0.0325
    assert np.isnan(df.loc["1926-07", "Soda"])  # -99.99 sentinel
    assert np.isnan(df.loc["1926-08", "Soda"])  # -999 sentinel
    assert df.loc["1926-09", "Soda"] == 0.015


def test_read_zip_text_and_vintage(tmp_path):
    path = tmp_path / "x.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("x.csv", FIXTURE)
    text = read_zip_text(path)
    assert data_vintage(text) == "This file was created using the 202608 CRSP database."
    assert parse_monthly_table(text).shape == (3, 3)


def test_factor_file_layout():
    text = io.StringIO()
    text.write("Header line\n\n,Mkt-RF,SMB,HML,RF\n196307, -0.39, -0.48, -0.84, 0.27\n")
    text.write("\n Annual Factors: January-December \n,Mkt-RF,SMB,HML,RF\n  1964, 1, 2, 3, 4\n")
    df = parse_monthly_table(text.getvalue())
    assert list(df.columns) == ["Mkt-RF", "SMB", "HML", "RF"]
    assert len(df) == 1
    assert np.isclose(df.iloc[0]["RF"], 0.0027)
