
from utils import load_csv, show_basic_info

if __name__ == "__main__":
    # Load the data
    df = load_csv("plant_dataset.csv")
    
    # Show basic info
    show_basic_info(df)
    
    # Show unique values in a column
    print("Unique priorities:", df['maintenance_priority'].unique())
    print("Priority counts:\n", df['maintenance_priority'].value_counts())
