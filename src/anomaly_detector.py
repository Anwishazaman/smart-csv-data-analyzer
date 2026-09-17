"""Use Isolation Forest to find unusual-looking employee rows.

The result is only a shortlist for inspection. It does not prove that a
record is wrong.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest


DEFAULT_FEATURES = ["age", "salary", "years_experience", "performance_score"]


def detect_anomalies(df, features=None, contamination=0.05, random_state=42):
    """
    Runs Isolation Forest on the given numeric features and adds two new
    columns to the dataframe:

      anomaly        -> 1 (normal) or -1 (potential anomaly), raw sklearn output
      anomaly_status -> "Normal" / "Potential Anomaly", human readable version

    contamination is basically "what % of the data do we expect to be
    anomalies" - 0.05 (5%) is a reasonable starting guess for this kind
    of dataset, can be tuned if needed.

    Returns (df_with_anomalies, list_of_features_actually_used) or
    (None, None) if there wasn't enough usable data.
    """
    df = df.copy()

    if features is None:
        features = DEFAULT_FEATURES

    usable_features = [
        col for col in features
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]

    if len(usable_features) < 2:
        print("Not enough numeric columns available to run anomaly detection.")
        print(f"(looked for: {features}, found usable: {usable_features})")
        return None, None

    feature_data = df[usable_features]

    valid_rows = feature_data.dropna()

    if len(valid_rows) < 10:
        print(f"Only {len(valid_rows)} complete rows available - that's too few to run anomaly detection reliably.")
        print("Try cleaning missing values first with --clean.")
        return None, None

    model = IsolationForest(contamination=contamination, random_state=random_state)
    predictions = model.fit_predict(valid_rows)

    df["anomaly"] = 1
    df.loc[valid_rows.index, "anomaly"] = predictions

    df["anomaly_status"] = df["anomaly"].map({1: "Normal", -1: "Potential Anomaly"})

    anomaly_count = (df["anomaly"] == -1).sum()
    print(f"\n--- Anomaly Detection ---")
    print(f"Features used: {usable_features}")
    print(f"Potential anomalies detected: {anomaly_count} out of {len(df)} rows")

    return df, usable_features


def show_anomalies(df, name_col="name"):
    """Prints out just the rows flagged as potential anomalies, for a quick look."""
    if "anomaly_status" not in df.columns:
        print("No anomaly data found - run detect_anomalies() first.")
        return

    flagged = df[df["anomaly_status"] == "Potential Anomaly"]
    if flagged.empty:
        print("No anomalies flagged.")
        return

    print(f"\n{len(flagged)} flagged record(s):")
    cols_to_show = [c for c in [name_col, "age", "department", "salary",
                                 "years_experience", "performance_score"] if c in flagged.columns]
    print(flagged[cols_to_show].to_string(index=False))
