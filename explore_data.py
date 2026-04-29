import pandas as pd

# Load DDoS attack traffic
ddos = pd.read_parquet('data/DDoS-Friday-no-metadata.parquet')

# Load normal (benign) traffic
benign = pd.read_parquet('data/Benign-Monday-no-metadata.parquet')

# See how many rows each has
print("DDoS traffic rows:", len(ddos))
print("Benign traffic rows:", len(benign))

# See what columns (features) exist
print("\nColumns:", ddos.columns.tolist())

# Peek at first 5 rows
print("\nFirst 5 rows of DDoS data:")
print(ddos.head())

# Check what labels exist in DDoS file
print("\nLabel types in DDoS file:")
print(ddos['Label'].value_counts())

# Check benign file
print("\nLabel types in Benign file:")
print(benign['Label'].value_counts())

# Check for missing values
print("\nMissing values in DDoS data:")
print(ddos.isnull().sum().sum())

print("\nMissing values in Benign data:")
print(benign.isnull().sum().sum())