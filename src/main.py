import argparse
import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import csv_loader
import data_analyzer
import data_cleaner
import anomaly_detector
import visualizer


REQUIRED_COLUMNS = ["employee_id", "name", "age", "department", "salary",
                    "years_experience", "performance_score"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Smart CSV Data Analyzer - load, clean, analyze and visualize a CSV file."
    )
    parser.add_argument("--file", default="data/sample_data.csv",
                         help="Path to the CSV file to analyze (default: data/sample_data.csv)")
    parser.add_argument("--analyze", action="store_true",
                         help="Only run the analysis/overview step, skip cleaning and the rest.")
    parser.add_argument("--clean", action="store_true",
                         help="Only run the cleaning step.")
    parser.add_argument("--detect-anomalies", action="store_true",
                         help="Only run anomaly detection (runs cleaning first, needed for the model).")
    parser.add_argument("--filter-column", help="Column name to filter on, e.g. salary")
    parser.add_argument("--filter-op", help="Comparison operator: > < >= <= == !=")
    parser.add_argument("--filter-value", help="Value to compare against")
    parser.add_argument("--no-charts", action="store_true",
                         help="Skip generating visualizations (useful if matplotlib is being annoying).")
    return parser.parse_args()


def run_full_pipeline(df, args):
    """Run the complete analysis and save the results."""
    report_lines = ["SMART CSV DATA ANALYZER", "=" * 40, ""]
    report_lines.append(f"Dataset: {os.path.basename(args.file)}")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("")

    overview = data_analyzer.dataset_overview(df)
    column_types = data_analyzer.identify_column_types(df)
    correlations = data_analyzer.correlation_analysis(df)
    outliers = data_analyzer.outlier_summary(df)
    report_lines.append(f"Rows: {overview['rows']}")
    report_lines.append(f"Columns: {overview['columns']}")
    report_lines.append("")
    report_lines.append("Missing Values:")
    if overview["missing_values"]:
        for col, count in overview["missing_values"].items():
            report_lines.append(f"  {col}: {count}")
    else:
        report_lines.append("  None")
    report_lines.append(f"\nDuplicate rows found: {overview['duplicate_rows']}")
    report_lines.append("\nColumn Types:")
    report_lines.append(f"  Numeric: {column_types['numeric']}")
    report_lines.append(f"  Categorical: {column_types['categorical']}")
    report_lines.append("\nPossible Outliers (IQR method):")
    for col, count in outliers.items():
        report_lines.append(f"  {col}: {count}")
    if correlations is not None:
        report_lines.append("\nCorrelation Matrix:")
        report_lines.append(correlations.to_string())

    missing_cols = csv_loader.check_required_columns(df, REQUIRED_COLUMNS)
    if missing_cols:
        print(f"\nWarning: dataset is missing expected columns: {missing_cols}")
        print("Some steps below may not work correctly.")

    cleaned_df, clean_summary = data_cleaner.clean_dataset(
        df, numeric_columns=column_types["numeric"],
        categorical_columns=column_types["categorical"]
    )
    data_cleaner.print_cleaning_summary(clean_summary)

    report_lines.append("\nCleaning Performed:")
    report_lines.append(f"  Duplicate rows removed: {clean_summary['duplicates_removed']}")
    for col, count in clean_summary["numeric_filled"].items():
        report_lines.append(f"  Missing {col} values filled: {count}")
    for col, count in clean_summary["categorical_filled"].items():
        report_lines.append(f"  Missing {col} values filled: {count}")

    stats = data_analyzer.descriptive_stats(cleaned_df)
    dept_summary = data_analyzer.department_summary(cleaned_df)

    if stats is not None:
        report_lines.append("\nKey Statistics (numeric columns):")
        report_lines.append(stats.round(2).to_string())

    if dept_summary is not None and "avg_salary" in dept_summary:
        report_lines.append("\nAverage Salary by Department:")
        report_lines.append(dept_summary["avg_salary"].to_string())

    result_df, used_features = anomaly_detector.detect_anomalies(cleaned_df)
    generated_files = []

    if result_df is not None:
        anomaly_detector.show_anomalies(result_df)
        anomaly_count = (result_df["anomaly"] == -1).sum()
        report_lines.append(f"\nAnomaly Detection:")
        report_lines.append(f"  Features used: {used_features}")
        report_lines.append(f"  Potential anomalies detected: {anomaly_count}")

        os.makedirs("output", exist_ok=True)
        analyzed_path = os.path.join("output", "analyzed_data.csv")
        result_df.to_csv(analyzed_path, index=False)
        print(f"\nSaved analyzed dataset to {analyzed_path}")
        generated_files.append(analyzed_path)
        final_df = result_df
    else:
        final_df = cleaned_df

    if args.filter_column and args.filter_op and args.filter_value is not None:
        filtered = data_analyzer.filter_data(final_df, args.filter_column, args.filter_op, args.filter_value)
        if filtered is not None:
            os.makedirs("output", exist_ok=True)
            filtered_path = os.path.join("output", "filtered_data.csv")
            filtered.to_csv(filtered_path, index=False)
            print(f"Saved filtered results to {filtered_path}")
            generated_files.append(filtered_path)

    if not args.no_charts:
        chart_paths = visualizer.generate_all_visualizations(final_df)
        generated_files.extend(chart_paths)

    report_lines.append("\nGenerated Files:")
    for f in generated_files:
        report_lines.append(f"  {f}")

    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "analysis_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\nAnalysis report written to {report_path}")


def main():
    args = parse_args()

    df = csv_loader.load_csv(args.file)
    if df is None:
        sys.exit(1)

    if args.analyze:
        data_analyzer.dataset_overview(df)
        data_analyzer.identify_column_types(df)
        data_analyzer.descriptive_stats(df)
        data_analyzer.correlation_analysis(df)
        data_analyzer.outlier_summary(df)
        data_analyzer.department_summary(df)
        return

    if args.clean:
        column_types = data_analyzer.identify_column_types(df)
        cleaned_df, summary = data_cleaner.clean_dataset(
            df,
            numeric_columns=column_types["numeric"],
            categorical_columns=column_types["categorical"]
        )
        data_cleaner.print_cleaning_summary(summary)
        os.makedirs("output", exist_ok=True)
        out_path = os.path.join("output", "cleaned_data.csv")
        cleaned_df.to_csv(out_path, index=False)
        print(f"\nSaved cleaned dataset to {out_path}")
        return

    if args.detect_anomalies:
        column_types = data_analyzer.identify_column_types(df)
        cleaned_df, _ = data_cleaner.clean_dataset(
            df,
            numeric_columns=column_types["numeric"],
            categorical_columns=column_types["categorical"]
        )
        result_df, _ = anomaly_detector.detect_anomalies(cleaned_df)
        if result_df is not None:
            anomaly_detector.show_anomalies(result_df)
            os.makedirs("output", exist_ok=True)
            out_path = os.path.join("output", "analyzed_data.csv")
            result_df.to_csv(out_path, index=False)
            print(f"\nSaved results to {out_path}")
        return

    run_full_pipeline(df, args)


if __name__ == "__main__":
    main()
