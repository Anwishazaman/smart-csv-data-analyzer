"""Answer questions about a dataset with an optional OpenAI model."""

import json
import os

import pandas as pd

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


def make_dataset_context(dataframe, overview, column_types, correlations, outliers):
    """Create a small profile to send to the language model."""
    numeric_summary = dataframe.select_dtypes(include="number").describe().round(2)
    missing = dataframe.isna().sum()
    missing = missing[missing > 0].to_dict()

    context = {
        "shape": {"rows": len(dataframe), "columns": len(dataframe.columns)},
        "columns": list(dataframe.columns),
        "numeric_columns": column_types["numeric"],
        "categorical_columns": column_types["categorical"],
        "missing_values": {column: int(count) for column, count in missing.items()},
        "duplicate_rows": overview["duplicate_rows"],
        "outliers_by_numeric_column": outliers,
        "numeric_summary": numeric_summary.to_dict(),
        "correlations": correlations.round(2).to_dict() if correlations is not None else {},
    }
    return json.dumps(context, indent=2, default=str)


def ask_dataset(question, dataset_context):
    """Ask the configured language model a question about the dataset profile."""
    if OpenAI is None:
        raise ValueError("The openai package is not installed. Run: pip install openai")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set the OPENAI_API_KEY environment variable to use the assistant.")

    client = OpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    response = client.chat.completions.create(
        model=model,
        temperature=0.2,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a careful data analysis assistant. Answer only from the "
                    "provided dataset profile. If the profile does not contain enough "
                    "information, say so. Use percentages and column names when useful. "
                    "Do not claim that correlation proves causation."
                ),
            },
            {
                "role": "user",
                "content": f"Dataset profile:\n{dataset_context}\n\nQuestion: {question}",
            },
        ],
    )
    return response.choices[0].message.content
