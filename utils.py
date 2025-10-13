# utils.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def load_csv(path):
    """Try to load a CSV file with common encodings."""
    for enc in ["utf-8", "latin1"]:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"Loaded with encoding {enc}")
            return df
        except Exception:
            continue
    raise Exception("Could not load CSV file.")

def show_basic_info(df):
    """Print basic info about the DataFrame."""
    print("Rows and columns:", df.shape)
    print("Column types:\n", df.dtypes)
    print("Missing values:\n", df.isnull().sum())
    print("Summary stats:\n", df.describe(include='all'))

def save_model(model, filename):
    """Save a model to a file."""
    joblib.dump(model, filename)

def load_model(filename):
    """Load a model from a file."""
    return joblib.load(filename)

def label_encode_columns(df, columns):
    """Label encode the given columns."""
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le
    return df, encoders

# Example usage:
# df = load_csv("mydata.csv")
# show_basic_info(df)
