import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load cleaned data
print("Loading data...")
df = pd.read_parquet('data/cleaned_data.parquet')

# Separate features and label
X = df.drop('Label', axis=1)
y = df['Label']

# Split into training and testing
# 80% train, 20% test
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Build the ML model
print("\nTraining model... (this may take 1-2 minutes)")
model = RandomForestClassifier(
    n_estimators=100,    # 100 decision trees
    random_state=42,
    n_jobs=-1            # use all CPU cores = faster
)

model.fit(X_train, y_train)
print("✅ Model trained!")

# Test the model
print("\nTesting model...")
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Accuracy: {accuracy * 100:.2f}%")

# Detailed report
print("\nDetailed Report:")
print(classification_report(y_test, y_pred, 
      target_names=['Benign', 'DDoS']))

# Save the model
with open('models/ddos_model.pkl', 'wb') as f:
    pickle.dump(model, f)
print("\n✅ Model saved to models/ddos_model.pkl")