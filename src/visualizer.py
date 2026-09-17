import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")

OUTPUT_DIR = "visualizations"


def _save(fig, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved chart: {path}")
    return path


def plot_salary_distribution(df, salary_col="salary"):
    if salary_col not in df.columns:
        print(f"Skipping salary distribution - no '{salary_col}' column.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(df[salary_col].dropna(), bins=20, kde=True, ax=ax, color="steelblue")
    ax.set_title("Salary Distribution")
    ax.set_xlabel("Salary")
    ax.set_ylabel("Number of Employees")
    return _save(fig, "salary_distribution.png")


def plot_department_distribution(df, department_col="department"):
    if department_col not in df.columns:
        print(f"Skipping department distribution - no '{department_col}' column.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[department_col].value_counts()
    sns.barplot(x=counts.index, y=counts.values, ax=ax, hue=counts.index, legend=False, palette="viridis")
    ax.set_title("Employees by Department")
    ax.set_xlabel("Department")
    ax.set_ylabel("Number of Employees")
    plt.xticks(rotation=30, ha="right")
    return _save(fig, "department_distribution.png")


def plot_salary_vs_experience(df, salary_col="salary", exp_col="years_experience"):
    if salary_col not in df.columns or exp_col not in df.columns:
        print("Skipping salary vs experience chart - missing required columns.")
        return None

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(data=df, x=exp_col, y=salary_col, ax=ax, alpha=0.7, color="darkorange")
    ax.set_title("Salary vs Years of Experience")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    return _save(fig, "salary_vs_experience.png")


def plot_salary_boxplot(df, salary_col="salary"):
    if salary_col not in df.columns:
        print(f"Skipping salary boxplot - no '{salary_col}' column.")
        return None

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.boxplot(y=df[salary_col].dropna(), ax=ax, color="lightcoral")
    ax.set_title("Salary Boxplot (Outlier Check)")
    ax.set_ylabel("Salary")
    return _save(fig, "salary_boxplot.png")


def plot_correlation_heatmap(df):
    """Save a heatmap showing relationships between numeric columns."""
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        print("Skipping correlation heatmap - not enough numeric columns.")
        return None

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(numeric_df.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    ax.set_title("Correlation Between Numeric Columns")
    return _save(fig, "correlation_heatmap.png")


def generate_all_visualizations(df):
    """Runs all the chart functions and returns a list of the files that got created."""
    print("\n--- Generating Visualizations ---")
    generated = []
    for path in [
        plot_salary_distribution(df),
        plot_department_distribution(df),
        plot_salary_vs_experience(df),
        plot_salary_boxplot(df),
        plot_correlation_heatmap(df),
    ]:
        if path:
            generated.append(path)
    return generated
