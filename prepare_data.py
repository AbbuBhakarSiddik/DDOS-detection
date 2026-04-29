import pandas as pd

# Load both files
ddos = pd.read_parquet('data/DDoS-Friday-no-metadata.parquet')
benign = pd.read_parquet('data/Benign-Monday-no-metadata.parquet')

# Combine them into one dataset
df = pd.concat([ddos, benign], ignore_index=True)

print("Total records:", len(df))
print("\nLabel counts:")
print(df['Label'].value_counts())

# Convert Label to numbers
# Benign = 0, DDoS = 1
df['Label'] = df['Label'].map({'Benign': 0, 'DDoS': 1})

print("\nAfter converting labels to numbers:")
print(df['Label'].value_counts())

# Separate features (X) and target (y)
X = df.drop('Label', axis=1)  # everything except Label
y = df['Label']               # only the Label column

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# Save cleaned data
df.to_parquet('data/cleaned_data.parquet', index=False)
print("\n✅ Cleaned data saved to data/cleaned_data.parquet")