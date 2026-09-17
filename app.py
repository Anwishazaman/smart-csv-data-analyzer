import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from src import data_analyzer
from src import data_cleaner
from src import assistant
from src import ml_models
from src import visualizer


st.set_page_config(page_title="Smart CSV Analyzer", layout="wide")
st.title("Smart CSV Data Analyzer")
st.write("Upload a CSV file to inspect its quality and get a quick analysis.")

uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

if uploaded_file is None:
    st.info("Choose a CSV file to begin.")
    st.stop()

try:
    dataframe = pd.read_csv(uploaded_file)
except Exception as error:
    st.error(f"The file could not be read: {error}")
    st.stop()

if dataframe.empty:
    st.warning("The CSV file has no rows to analyze.")
    st.stop()

column_types = data_analyzer.identify_column_types(dataframe)
overview = data_analyzer.dataset_overview(dataframe)
correlations = data_analyzer.correlation_analysis(dataframe)
outliers = data_analyzer.outlier_summary(dataframe)

st.success(f"Loaded {len(dataframe)} rows and {len(dataframe.columns)} columns.")

summary_tab, quality_tab, charts_tab, ml_tab, assistant_tab, report_tab = st.tabs(
    ["Summary", "Data quality", "Charts", "Predict a column", "Ask about data", "Report"]
)

with summary_tab:
    first_column, second_column = st.columns(2)
    first_column.metric("Rows", overview["rows"])
    second_column.metric("Columns", overview["columns"])

    st.subheader("Column types")
    st.write("**Numeric:**", ", ".join(column_types["numeric"]) or "None")
    st.write("**Categorical:**", ", ".join(column_types["categorical"]) or "None")

    st.subheader("Statistical summary")
    statistics = data_analyzer.descriptive_stats(dataframe)
    if statistics is not None:
        st.dataframe(statistics)
    else:
        st.write("No numeric columns were found.")

    st.subheader("Correlation analysis")
    if correlations is not None:
        st.dataframe(correlations)
    else:
        st.write("There are not enough numeric columns for correlation analysis.")

with quality_tab:
    st.subheader("Missing values")
    missing_values = dataframe.isna().sum()
    missing_values = missing_values[missing_values > 0]
    if missing_values.empty:
        st.success("No missing values found.")
    else:
        missing_table = pd.DataFrame({"Missing values": missing_values})
        missing_table["Missing percent"] = (
            missing_table["Missing values"] / len(dataframe) * 100
        ).round(2)
        st.dataframe(missing_table)

    st.subheader("Duplicates")
    st.write(f"Duplicate rows found: **{overview['duplicate_rows']}**")

    st.subheader("Possible outliers")
    outlier_table = pd.DataFrame.from_dict(
        outliers, orient="index", columns=["Possible outliers"]
    )
    st.dataframe(outlier_table)

    invalid_values = []
    if "age" in dataframe.columns:
        invalid_values.append({
            "column": "age",
            "problem": "Values below 0 or above 100",
            "rows": int(((dataframe["age"] < 0) | (dataframe["age"] > 100)).sum()),
        })
    if "salary" in dataframe.columns:
        invalid_values.append({
            "column": "salary",
            "problem": "Values below 0",
            "rows": int((dataframe["salary"] < 0).sum()),
        })

    st.subheader("Simple value checks")
    if invalid_values:
        st.dataframe(pd.DataFrame(invalid_values))
    else:
        st.write("No employee-specific value checks could be applied to this file.")

with charts_tab:
    st.subheader("Visualizations")
    chart_functions = [
        ("Salary distribution", visualizer.plot_salary_distribution),
        ("Department distribution", visualizer.plot_department_distribution),
        ("Salary and experience", visualizer.plot_salary_vs_experience),
        ("Salary boxplot", visualizer.plot_salary_boxplot),
    ]

    for title, chart_function in chart_functions:
        figure, axis = plt.subplots(figsize=(8, 4))
        if title == "Salary distribution" and "salary" in dataframe.columns:
            sns_data = dataframe["salary"].dropna()
            axis.hist(sns_data, bins=20, color="steelblue")
            axis.set_xlabel("Salary")
            axis.set_ylabel("Number of employees")
        elif title == "Department distribution" and "department" in dataframe.columns:
            counts = dataframe["department"].value_counts()
            axis.bar(counts.index, counts.values, color="steelblue")
            axis.tick_params(axis="x", rotation=30)
            axis.set_ylabel("Number of employees")
        elif title == "Salary and experience" and {"salary", "years_experience"}.issubset(dataframe.columns):
            axis.scatter(dataframe["years_experience"], dataframe["salary"], alpha=0.7)
            axis.set_xlabel("Years of experience")
            axis.set_ylabel("Salary")
        elif title == "Salary boxplot" and "salary" in dataframe.columns:
            axis.boxplot(dataframe["salary"].dropna())
            axis.set_ylabel("Salary")
        else:
            plt.close(figure)
            continue
        axis.set_title(title)
        st.pyplot(figure)
        plt.close(figure)

    if correlations is not None:
        figure, axis = plt.subplots(figsize=(8, 6))
        image = axis.imshow(correlations, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_xticks(range(len(correlations.columns)), correlations.columns, rotation=45, ha="right")
        axis.set_yticks(range(len(correlations.index)), correlations.index)
        figure.colorbar(image, ax=axis)
        axis.set_title("Correlation heatmap")
        st.pyplot(figure)
        plt.close(figure)

with ml_tab:
    st.subheader("Predict this column")
    st.write(
        "Choose a column to predict. The app will decide whether this is "
        "classification or regression and compare several models."
    )
    target_column = st.selectbox("Target column", dataframe.columns)

    if st.button("Train models"):
        try:
            with st.spinner("Training models..."):
                ml_result = ml_models.train_and_compare(dataframe, target_column)

            st.write(f"Problem type: **{ml_result['problem_type']}**")
            st.write(f"Rows used for testing: **{ml_result['x_test_rows']}**")
            st.dataframe(ml_result["comparison"], use_container_width=True)
            st.success(f"Best model: {ml_result['best_name']}")

            if ml_result["feature_importance"] is not None:
                st.subheader("Feature importance")
                st.dataframe(ml_result["feature_importance"], use_container_width=True)
            else:
                st.info("This model does not provide feature importance values.")

            st.subheader("Why did this prediction happen?")
            st.write("SHAP shows which features pushed the first test prediction up or down.")
            try:
                explanation = ml_models.explain_prediction(
                    ml_result["best_model"], ml_result["test_features"]
                )
                prediction = explanation["prediction"]
                if ml_result["target_encoder"] is not None:
                    prediction = ml_result["target_encoder"].inverse_transform([prediction])[0]
                st.write(f"Prediction explained: **{prediction}**")
                st.write(f"Base model value: **{explanation['base_value']:.4f}**")
                st.dataframe(explanation["contributions"], use_container_width=True)
            except ValueError as error:
                st.warning(str(error))
        except ValueError as error:
            st.error(str(error))

with assistant_tab:
    st.subheader("Ask questions about your dataset")
    st.write("Ask about missing data, unusual values, statistics, or correlations.")
    question = st.text_input(
        "Your question",
        placeholder="Which columns have the most missing data?",
    )

    if st.button("Ask assistant"):
        if not question.strip():
            st.warning("Type a question first.")
        else:
            dataset_context = assistant.make_dataset_context(
                dataframe, overview, column_types, correlations, outliers
            )
            try:
                with st.spinner("Thinking about the dataset..."):
                    answer = assistant.ask_dataset(question, dataset_context)
                st.write(answer)
            except ValueError as error:
                st.error(str(error))

with report_tab:
    cleaned_dataframe, cleaning_summary = data_cleaner.clean_dataset(
        dataframe,
        numeric_columns=column_types["numeric"],
        categorical_columns=column_types["categorical"],
    )

    report = [
        "SMART CSV DATA ANALYZER",
        "",
        f"Rows: {overview['rows']}",
        f"Columns: {overview['columns']}",
        f"Duplicate rows: {overview['duplicate_rows']}",
        f"Numeric columns: {column_types['numeric']}",
        f"Categorical columns: {column_types['categorical']}",
        "",
        "Missing values:",
    ]
    report.extend(f"  {column}: {count}" for column, count in missing_values.items())
    report.append("")
    report.append("Possible outliers:")
    report.extend(f"  {column}: {count}" for column, count in outliers.items())
    report.append("")
    report.append("Correlation matrix:")
    if correlations is None:
        report.append("  Not enough numeric columns")
    else:
        report.append(correlations.to_string())
    report.append("")
    report.append("Value checks:")
    if invalid_values:
        report.extend(
            f"  {item['column']}: {item['problem']} ({item['rows']} rows)"
            for item in invalid_values
        )
    else:
        report.append("  No employee-specific checks were applied")
    report.append("")
    report.append("Cleaning summary:")
    report.append(f"  Duplicates removed: {cleaning_summary['duplicates_removed']}")
    report.append(f"  Numeric values filled: {cleaning_summary['numeric_filled']}")
    report.append(f"  Categorical values filled: {cleaning_summary['categorical_filled']}")
    report_text = "\n".join(report)

    st.text_area("Generated quality report", report_text, height=350)
    st.download_button(
        "Download quality report",
        data=report_text,
        file_name="analysis_report.txt",
        mime="text/plain",
    )
    st.download_button(
        "Download cleaned CSV",
        data=cleaned_dataframe.to_csv(index=False).encode("utf-8"),
        file_name="cleaned_data.csv",
        mime="text/csv",
    )
