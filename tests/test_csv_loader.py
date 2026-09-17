"""
Basic tests for csv_loader.py. Nothing fancy, just checking the common
cases actually work the way I expect.
"""

import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from csv_loader import load_csv, check_required_columns


def test_load_valid_csv(tmp_path):
    file_path = tmp_path / "test.csv"
    file_path.write_text("a,b,c\n1,2,3\n4,5,6\n")

    df = load_csv(str(file_path))

    assert df is not None
    assert df.shape == (2, 3)
    assert list(df.columns) == ["a", "b", "c"]


def test_missing_file_returns_none(capsys):
    df = load_csv("this/path/does/not/exist.csv")

    assert df is None
    captured = capsys.readouterr()
    assert "could not find" in captured.out.lower()


def test_empty_csv_returns_none(tmp_path):
    file_path = tmp_path / "empty.csv"
    file_path.write_text("")

    df = load_csv(str(file_path))
    assert df is None


def test_check_required_columns_all_present():
    df = pd.DataFrame({"a": [1], "b": [2]})
    missing = check_required_columns(df, ["a", "b"])
    assert missing == []


def test_check_required_columns_some_missing():
    df = pd.DataFrame({"a": [1]})
    missing = check_required_columns(df, ["a", "b", "c"])
    assert missing == ["b", "c"]
