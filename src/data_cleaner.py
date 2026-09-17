"""Small cleaning functions for duplicates and missing values."""

import pandas as pd


def remove_duplicates(df):
    """Drops exact duplicate rows, returns (cleaned_df, number_removed)."""
    before = len(df)
    cleaned = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(cleaned)
    return cleaned, removed


def fill_missing_numeric(df, columns=None, strategy="median"):
    """
    Fills missing values in numeric columns.

    strategy: "median" or "mean". Went with median as the default since
    it's less sensitive to the crazy outlier salaries in this dataset -
    a single $999,000 typo would drag the mean way up.

    Returns (cleaned_df, dict of {column: number_filled}).
    """
    df = df.copy()
    filled_counts = {}

    if columns is None:
        columns = df.select_dtypes(include="number").columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        if strategy == "mean":
            fill_value = df[col].mean()
        else:
            fill_value = df[col].median()

        df[col] = df[col].fillna(fill_value)
        filled_counts[col] = int(missing_count)

    return df, filled_counts


def fill_missing_categorical(df, columns=None):

    df = df.copy()
    filled_counts = {}

    if columns is None:
        columns = df.select_dtypes(include="object").columns.tolist()

    for col in columns:
        if col not in df.columns:
            continue
        missing_count = df[col].isnull().sum()
        if missing_count == 0:
            continue

        mode_values = df[col].mode()
        if mode_values.empty:
            continue

        fill_value = mode_values[0]
        df[col] = df[col].fillna(fill_value)
        filled_counts[col] = int(missing_count)

    return df, filled_counts


def validate_dtypes(df, expected_types):
    """
    Basic check that columns are the types we expect them to be, e.g.
    {"age": "number", "name": "text"}. Doesn't try to fix anything, just
    reports mismatches so you know something's off before you run stats
    on it.
    """
    issues = []
    for col, expected in expected_types.items():
        if col not in df.columns:
            issues.append(f"Column '{col}' not found in dataset.")
            continue

        is_numeric = pd.api.types.is_numeric_dtype(df[col])
        if expected == "number" and not is_numeric:
            issues.append(f"Column '{col}' expected to be numeric but isn't.")
        elif expected == "text" and is_numeric:
            issues.append(f"Column '{col}' expected to be text but looks numeric.")

    return issues


def clean_dataset(df, numeric_columns=None, categorical_columns=None):
    """
    Runs the full cleaning pipeline (dupes -> numeric fill -> categorical
    fill) and returns the cleaned dataframe plus a summary dict, so
    main.py doesn't need to know the individual steps.
    """
    summary = {}

    df, removed = remove_duplicates(df)
    summary["duplicates_removed"] = removed

    df, numeric_filled = fill_missing_numeric(df, columns=numeric_columns)
    summary["numeric_filled"] = numeric_filled

    df, categorical_filled = fill_missing_categorical(df, columns=categorical_columns)
    summary["categorical_filled"] = categorical_filled

    return df, summary


def print_cleaning_summary(summary):
    print("\n--- Cleaning Summary ---")
    print(f"Duplicate rows removed: {summary.get('duplicates_removed', 0)}")

    numeric_filled = summary.get("numeric_filled", {})
    if numeric_filled:
        for col, count in numeric_filled.items():
            print(f"Missing {col} values filled: {count}")

    categorical_filled = summary.get("categorical_filled", {})
    if categorical_filled:
        for col, count in categorical_filled.items():
            print(f"Missing {col} values filled: {count}")

    if not numeric_filled and not categorical_filled:
        print("No missing values needed filling.")
