import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

def preprocess_data(df, target_col=None):
    """
    Fresh preprocessing per request:
    - Keep numeric columns unchanged in scale (only fill NaNs with median)
    - Convert categorical/text columns to numeric using best-fit encoding per cardinality
      • One-Hot for low-cardinality (<= 15 unique)
      • Frequency encoding for high-cardinality (> 15 unique)
    - Parse dates into numeric (year/month/dayofweek + relative days) and drop raw date strings
    - Drop obvious identifiers/noisy columns (IDs, names, emails, URLs, remarks)
    Returns: processed DataFrame, encoders dict
    """

    df = df.copy()

    # Keep target aside if provided
    target_series = None
    if target_col is not None and target_col in df.columns:
        target_series = df[target_col]

    # Drop / Ignore noisy identifiers and free-text
    drop_cols_default = [
        'asset_id',
        'responsible_person_id',
        'responsible_person_name',
        'contact_email',
        'manufacturer_url',
        'remarks',
    ]
    df = df.drop(columns=[c for c in drop_cols_default if c in df.columns], errors='ignore')

    # Date Feature Extraction
    date_cols = ['install_date', 'last_maintenance_date', 'next_due_date']
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce', dayfirst=True)

    # Basic date parts
    if 'install_date' in df.columns:
        df['install_year'] = df['install_date'].dt.year
        df['install_month'] = df['install_date'].dt.month
        df['install_dow'] = df['install_date'].dt.dayofweek
    if 'last_maintenance_date' in df.columns:
        df['last_maint_year'] = df['last_maintenance_date'].dt.year
        df['last_maint_month'] = df['last_maintenance_date'].dt.month
        df['last_maint_dow'] = df['last_maintenance_date'].dt.dayofweek
    if 'next_due_date' in df.columns:
        df['next_due_year'] = df['next_due_date'].dt.year
        df['next_due_month'] = df['next_due_date'].dt.month
        df['next_due_dow'] = df['next_due_date'].dt.dayofweek

    # Relative durations (using today)
    today = pd.Timestamp.today().normalize()
    if 'install_date' in df.columns:
        df['age_days'] = (today - df['install_date']).dt.days
    if 'last_maintenance_date' in df.columns:
        df['days_since_last_maint'] = (today - df['last_maintenance_date']).dt.days
    if 'next_due_date' in df.columns:
        df['days_until_next_due'] = (df['next_due_date'] - today).dt.days

    # Drop raw date columns
    df = df.drop(columns=[c for c in date_cols if c in df.columns], errors='ignore')

    encoders = {}

    # Identify numeric columns explicitly (ensure proper dtype but no scaling)
    explicit_numeric = [
        'runtime_hours', 'vibration_level', 'temperature', 'pressure',
        'failure_probability', 'model_release_year'
    ]
    for c in explicit_numeric:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Decide categorical candidates (object dtype) excluding target
    cat_cols = df.select_dtypes(include=['object']).columns.tolist()
    if target_series is not None and target_col in cat_cols:
        cat_cols.remove(target_col)

    # Split by cardinality threshold
    low_card_cols = [c for c in cat_cols if df[c].nunique(dropna=True) <= 15]
    high_card_cols = [c for c in cat_cols if c not in low_card_cols]

    # Frequency encode high-cardinality
    for c in high_card_cols:
        freq = df[c].fillna('NA').value_counts(normalize=True)
        df[c] = df[c].fillna('NA').map(freq).fillna(0.0)
        encoders[f'freq_{c}'] = freq

    # One-Hot encode low-cardinality
    if low_card_cols:
        ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        ohe_arr = ohe.fit_transform(df[low_card_cols])
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe.get_feature_names_out(low_card_cols), index=df.index)
        df = df.drop(columns=low_card_cols)
        df = pd.concat([df, ohe_df], axis=1)
        encoders['onehot'] = {'encoder': ohe, 'columns': low_card_cols}

    # Numeric imputation (no scaling)
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if target_series is not None and target_col in numeric_cols:
        numeric_cols = [c for c in numeric_cols if c != target_col]

    if numeric_cols:
        num_imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        encoders['num_imputer'] = {'imputer': num_imputer, 'columns': numeric_cols}

    # Feature example (kept simple; uses existing numeric scale)
    if 'vibration_level' in df.columns and 'temperature' in df.columns:
        df['vib_temp'] = df['vibration_level'] * df['temperature']

    # Reattach target if provided
    if target_series is not None:
        df[target_col] = target_series

    encoders['feature_columns'] = [c for c in df.columns if c != target_col]
    return df, encoders


# 🧪 Example test run (only runs when this file is executed directly)
if __name__ == "__main__":
    try:
        df = pd.read_csv("plant_dataset.csv", encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv("plant_dataset.csv", encoding='latin1')
    processed_df, encoders = preprocess_data(df, target_col=None)
    print(" Preprocessing complete. Here's a sample:")
    print(processed_df.head())
    processed_df.to_csv("processed_data.csv", index=False)
    print(" Saved to processed_data.csv")
