"""Functions for looking at the data without changing it."""

import pandas as pd


def dataset_overview(df):
    """Prints out the basic shape/structure of the dataset and returns it as a dict too, in case main.py wants it for the report."""

    print("\n--- Dataset Summary ---")
    rows, cols = df.shape
    print(f"Rows: {rows}")
    print(f"Columns: {cols}")

    print("\nColumns:")
    for col in df.columns:
        print(f"  {col} ({df[col].dtype})")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    print("\nMissing values:")
    if missing.empty:
        print("  None - dataset is complete.")
    else:
        for col, count in missing.items():
            print(f"  {col}: {count}")

    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes}")

    overview = {
        "rows": rows,
        "columns": cols,
        "column_names": list(df.columns),
        "missing_values": missing.to_dict(),
        "duplicate_rows": int(dupes),
    }
    return overview


def identify_column_types(df):
    """Return the numeric and text columns found in the dataframe."""
    numeric = df.select_dtypes(include="number").columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    print("\n--- Column Types ---")
    print(f"Numeric columns: {numeric}")
    print(f"Categorical columns: {categorical}")

    return {"numeric": numeric, "categorical": categorical}


def descriptive_stats(df):
    """
    Basic descriptive statistics for numeric columns using describe().
    Returns the describe() output so it can be reused elsewhere (e.g in
    the report) without printing again.
    """
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        print("\nNo numeric columns found, skipping statistics.")
        return None

    print("\n--- Descriptive Statistics (numeric columns) ---")
    stats = numeric_df.describe()
    print(stats)
    return stats


def correlation_analysis(df):
    """Calculate correlations between numeric columns."""
    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        print("\nNot enough numeric columns for correlation analysis.")
        return None

    correlations = numeric_df.corr().round(2)
    print("\n--- Correlation Analysis ---")
    print(correlations)
    return correlations


def outlier_summary(df):
    """Count possible outliers in each numeric column using the IQR rule."""
    outliers = {}

    for column in df.select_dtypes(include="number").columns:
        values = df[column].dropna()
        if values.empty:
            continue

        first_quartile = values.quantile(0.25)
        third_quartile = values.quantile(0.75)
        iqr = third_quartile - first_quartile
        lower_limit = first_quartile - 1.5 * iqr
        upper_limit = third_quartile + 1.5 * iqr
        count = int(((values < lower_limit) | (values > upper_limit)).sum())
        outliers[column] = count

    print("\n--- Possible Outliers (IQR method) ---")
    for column, count in outliers.items():
        print(f"{column}: {count}")

    return outliers


def department_summary(df, department_col="department", salary_col="salary"):
    """
    Group-by style summary: how many people per department, and average
    salary per department. Made the column names parameters instead of
    hardcoding them so this could technically work on a different dataset
    with different column names, though I mostly just use the defaults.
    """
    if department_col not in df.columns:
        print(f"\nCan't do department summary - no '{department_col}' column.")
        return None

    print(f"\n--- Employees per {department_col} ---")
    counts = df[department_col].value_counts()
    print(counts)

    if salary_col in df.columns:
        print(f"\n--- Average {salary_col} by {department_col} ---")
        avg_salary = df.groupby(department_col)[salary_col].mean().round(2).sort_values(ascending=False)
        print(avg_salary)
        return {"counts": counts, "avg_salary": avg_salary}

    return {"counts": counts}


def filter_data(df, column, operator, value):
    """
    Generic filter function so we're not hardcoding "salary > 100000" as
    a one-off. Supports the basic comparison operators.

    column:   name of the column to filter on
    operator: one of '>', '<', '>=', '<=', '==', '!='
    value:    value to compare against (gets cast to match column dtype
              when it's numeric)
    """
    if column not in df.columns:
        print(f"Error: column '{column}' not found in dataset.")
        return None

    valid_ops = [">", "<", ">=", "<=", "==", "!="]
    if operator not in valid_ops:
        print(f"Error: '{operator}' isn't a supported operator. Use one of {valid_ops}")
        return None

    col_series = df[column]

    if pd.api.types.is_numeric_dtype(col_series):
        try:
            value = float(value)
        except ValueError:
            print(f"Error: '{value}' isn't a valid number for column '{column}'.")
            return None

    if operator == ">":
        result = df[col_series > value]
    elif operator == "<":
        result = df[col_series < value]
    elif operator == ">=":
        result = df[col_series >= value]
    elif operator == "<=":
        result = df[col_series <= value]
    elif operator == "==":
        result = df[col_series == value]
    else:
        result = df[col_series != value]

    print(f"\nFilter: {column} {operator} {value} -> {len(result)} matching rows")
    return result
