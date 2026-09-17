"""Load the CSV file and do a few basic checks before using it."""

import os
import pandas as pd


def load_csv(file_path):
    """
    Loads a CSV file into a pandas DataFrame.

    Does some basic sanity checks first because pandas errors when
    something goes wrong are not exactly beginner friendly - you get
    a huge traceback instead of a plain "hey your file is empty".

    Returns the DataFrame, or None if it couldn't be loaded (in which
    case an error message has already been printed).
    """

    if not os.path.exists(file_path):
        print(f"Error: could not find a file at '{file_path}'.")
        print("Double check the path and try again.")
        return None

    if not os.path.isfile(file_path):
        print(f"Error: '{file_path}' is not a file (looks like a folder?).")
        return None

    if not file_path.lower().endswith(".csv"):
        print(f"Warning: '{file_path}' doesn't have a .csv extension. Trying to load it anyway.")

    try:
        df = pd.read_csv(file_path)
    except pd.errors.EmptyDataError:
        print(f"Error: '{file_path}' is empty. Nothing to analyze here.")
        return None
    except pd.errors.ParserError as e:
        print(f"Error: '{file_path}' doesn't look like a valid CSV file.")
        print(f"(pandas said: {e})")
        return None
    except UnicodeDecodeError:
        print(f"Error: couldn't read '{file_path}' - encoding issue.")
        print("Try saving the file as UTF-8 and run this again.")
        return None
    except Exception as e:
        print(f"Something went wrong loading '{file_path}': {e}")
        return None

    if df.empty:
        print(f"Warning: '{file_path}' loaded fine but has 0 rows.")
        return None

    if len(df.columns) == 0:
        print(f"Error: '{file_path}' has no columns, can't do anything with this.")
        return None

    print(f"Loaded '{file_path}' successfully - {df.shape[0]} rows, {df.shape[1]} columns.")
    return df


def check_required_columns(df, required_columns):
    """
    Quick helper to check the dataset actually has the columns we need
    before we try to run analysis on them. Returns list of missing columns
    (empty list means everything is fine).
    """
    missing = [col for col in required_columns if col not in df.columns]
    return missing
