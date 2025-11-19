import pandas as pd

pd.set_option("display.max_colwidth", None)  # don't truncate column values
pd.set_option("display.max_columns", None)  # show all columns
pd.set_option("display.width", None)  # don't wrap awkwardly

# Load the parquet dataset
df = pd.read_parquet("data/processed/final_dataset.parquet")

df.head(100).to_csv(
    "data/trial_output.csv", index=False
)  # Save to CSV for better viewing
