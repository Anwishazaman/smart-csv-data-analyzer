import os
import sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from data_analyzer import dataset_overview, filter_data


def sample_df():
    return pd.DataFrame({
        "name": ["Alice", "Bob", "Charlie", "Bob"],
        "age": [25, 30, 35, 30],
        "salary": [50000, 60000, 70000, 60000],
        "department": ["Engineering", "Sales", "Engineering", "Sales"],
    })


def test_dataset_overview_shape():
    df = sample_df()
    overview = dataset_overview(df)
    assert overview["rows"] == 4
    assert overview["columns"] == 4


def test_dataset_overview_detects_duplicates():
    df = sample_df()
    overview = dataset_overview(df)
    assert overview["duplicate_rows"] == 1


def test_dataset_overview_no_missing_values():
    df = sample_df()
    overview = dataset_overview(df)
    assert overview["missing_values"] == {}


def test_dataset_overview_detects_missing_values():
    df = sample_df()
    df.loc[0, "salary"] = None
    overview = dataset_overview(df)
    assert overview["missing_values"]["salary"] == 1


def test_filter_data_greater_than():
    df = sample_df()
    result = filter_data(df, "salary", ">", 55000)
    assert len(result) == 3
    assert (result["salary"] > 55000).all()


def test_filter_data_equals_string_column():
    df = sample_df()
    result = filter_data(df, "department", "==", "Engineering")
    assert len(result) == 2


def test_filter_data_invalid_column():
    df = sample_df()
    result = filter_data(df, "not_a_real_column", ">", 5)
    assert result is None


def test_filter_data_invalid_operator():
    df = sample_df()
    result = filter_data(df, "salary", "~=", 5)
    assert result is None
