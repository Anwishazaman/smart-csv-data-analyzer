# Smart CSV Data Analyzer

This is a small Python project made while learning pandas, matplotlib,
seaborn, scikit-learn, SHAP, and Streamlit.

The aim was to practice the usual steps in a data project in one place instead
of doing each step in separate notebook cells. The program takes an employee
CSV file, checks it, cleans a few common problems, prints some basic results,
and saves charts and files.

## What it does

Point it at a CSV file (or just use the sample one included) and it will:

- Load a CSV and print a useful error when it cannot be opened
- Show the number of rows, columns, missing values, and duplicate rows
- Detect numeric and categorical columns automatically
- Print basic statistics for numeric columns
- Show employee counts and average salary by department
- Remove duplicate rows and fill missing values
- Find possible outliers using Isolation Forest and the IQR method
- Show correlations between numeric columns
- Filter rows with comparisons such as `salary > 100000`
- Use Isolation Forest to find records that look unusual
- Save simple charts, including a correlation heatmap, and a text report
- Compare machine-learning models for a selected prediction column

## Trying a prediction

The Streamlit app has a `Predict a column` tab. Choose a column and click
`Train models`. The app decides whether the target is classification or
regression, fills missing feature values, encodes text columns, splits the
data into training and test sets, and compares the models.

For classification it uses Logistic Regression, Random Forest, and XGBoost.
For regression it uses Linear Regression, Random Forest, Gradient Boosting,
and XGBoost. It shows the scores, selects the best score from the test set,
and displays feature importance when the selected model supports it.

The prediction result also includes SHAP explanations. These show which
processed features increased or decreased the selected prediction for a
held-out test row.

## Asking questions about the data

The `Ask about data` tab uses an OpenAI model to answer questions about the
dataset profile. The app sends column names, data types, missing-value counts,
statistics, outlier counts, and correlations, rather than the whole CSV.

Set an API key before starting Streamlit:

```bash
$env:OPENAI_API_KEY="your-api-key"
streamlit run app.py
```

You can optionally choose a model with `OPENAI_MODEL`. The default is
`gpt-4o-mini`.

## Why I used an employee dataset
I used a fake employee dataset (`data/sample_data.csv`) to add some
problems on purpose. It contains missing values, duplicate rows, and unusual
values such as an incorrect-looking salary and age. This made it easier to
test the cleaning and anomaly detection parts.

Nothing in the CSV is real - names were randomly generated, this is not
anyone's actual data.

## Project structure

```
smart-csv-data-analyzer/
├── data/
│   └── sample_data.csv          sample dataset with intentional issues
├── output/                      cleaned/filtered/analyzed CSVs get saved here
├── reports/                     analysis_report.txt gets saved here
├── visualizations/              generated charts (.png) get saved here
├── src/
│   ├── main.py                  CLI entry point, ties everything together
│   ├── csv_loader.py            loading + validating the CSV
│   ├── data_analyzer.py         overview, stats, filtering
│   ├── data_cleaner.py          duplicate removal, missing value handling
│   ├── anomaly_detector.py      Isolation Forest based anomaly detection
│   ├── assistant.py              questions about the dataset profile
│   ├── ml_models.py              model training and SHAP explanations
│   └── visualizer.py            matplotlib/seaborn charts
├── app.py                        Streamlit browser application
├── tests/                       pytest tests for the core modules
├── requirements.txt
└── README.md
```

## Setup

```bash
git clone https://github.com/yourusername/smart-csv-data-analyzer.git
cd smart-csv-data-analyzer
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

Start the browser application:

```bash
streamlit run app.py
```

The command-line version is also available:

Run the full pipeline (analysis + cleaning + anomaly detection + charts + report)
on the included sample data:

```bash
python src/main.py --file data/sample_data.csv
```

Or use your own CSV - as long as it has similar columns
(`age`, `salary`, `department`, `years_experience`, `performance_score`)
things should mostly work, though the code tries to fail gracefully if
some columns aren't there.

```bash
python src/main.py --file path/to/your_data.csv
```

### Running just one step

If you don't want the whole pipeline, there are flags for the individual pieces:

```bash
python src/main.py --file data/sample_data.csv --analyze

python src/main.py --file data/sample_data.csv --clean

python src/main.py --file data/sample_data.csv --detect-anomalies
```

### Filtering

You can also filter records during the full pipeline run:

```bash
python src/main.py --file data/sample_data.csv --filter-column salary --filter-op ">" --filter-value 100000
```

This saves the matching rows to `output/filtered_data.csv`. Supported
operators are `>`, `<`, `>=`, `<=`, `==`, `!=`.

### Skipping charts

Matplotlib can be a pain on some machines (especially over SSH with no
display). If you just want the text output without generating PNGs:

```bash
python src/main.py --file data/sample_data.csv --no-charts
```

## Sample output

Running the default pipeline prints something like this to the terminal:

```
Loaded 'data/sample_data.csv' successfully - 123 rows, 8 columns.

--- Dataset Summary ---
Rows: 123
Columns: 8

Missing values:
  age: 2
  department: 3
  salary: 4

Duplicate rows: 3

--- Cleaning Summary ---
Duplicate rows removed: 3
Missing age values filled: 2
Missing department values filled: 3
Missing salary values filled: 4

--- Anomaly Detection ---
Features used: ['age', 'salary', 'years_experience', 'performance_score']
Potential anomalies detected: 6 out of 120 rows
```

...plus a written report in `reports/analysis_report.txt` and charts saved
in `visualizations/`.

## About the anomaly detection

This uses scikit-learn's `IsolationForest`, which is an **unsupervised**
model - it doesn't know what a "real" anomaly looks like, it just isolates
data points that are easier to separate from the rest based on the numeric
features it's given (age, salary, years of experience, performance score).

That means a row getting flagged as `Potential Anomaly` doesn't mean it's
definitely wrong or fraudulent. It could be:

- Genuinely bad data (a typo, like `999000` instead of `99000`)
- A real employee who's just unusual (very senior, unusually low tenure for their salary, etc.)

Treat the flags as "worth a second look", not a verdict. This is the same
kind of caveat you'd give with any unsupervised outlier detection - it's a
tool for narrowing down where to look, not an automatic judgment call.

## Running the tests

```bash
pytest
```

Tests cover the CSV loading (valid files, missing files, empty files) and
the core analysis functions (dataset overview, duplicate detection, missing
value detection, filtering). I kept them fairly basic on purpose since the
goal was to make sure the main logic actually works, not to hit 100%
coverage.

## Things I'd add if I kept working on this

- A config file (YAML/JSON) so column names aren't hardcoded to the
  employee dataset schema
- Support for more file types (Excel, JSON)
- A proper CLI subcommand structure instead of flags (`analyzer clean`,
  `analyzer detect` etc.) using something like `click`
- Letting the user tune the Isolation Forest's `contamination` parameter
  from the command line instead of it being fixed at 5%

## Notes

This was built as a learning project to get more comfortable with the
pandas -> matplotlib/seaborn -> scikit-learn workflow, model explanations,
and a small Streamlit interface. The data is analyzed in memory and there is
no database yet.
