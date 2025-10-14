
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import joblib

def load_csv(path):
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

# Data exploration functionality (merged from data_load_and_eda.py)
def explore_plant_data(csv_path="plant_dataset.csv"):  
    df = load_csv(csv_path)
    show_basic_info(df)
     # Show unique values in maintenance_priority column
    if 'maintenance_priority' in df.columns:
        print("Unique priorities:", df['maintenance_priority'].unique())
        print("Priority counts:\n", df['maintenance_priority'].value_counts())
    
    return df
if __name__ == "__main__":
    explore_plant_data()
