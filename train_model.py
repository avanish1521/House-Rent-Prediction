import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

print("Loading dataset...")

# Load dataset
df = pd.read_csv("House_Rent_Dataset.csv")

print("Dataset loaded successfully!")
print("Rows:", df.shape[0])
print("Columns:", df.shape[1])
print(df.head())

# Drop unnecessary columns
drop_cols = ["Posted On", "Point of Contact", "Area Locality"]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# Target
y = df["Rent"]

# Features
X = df.drop(columns=["Rent"])

# Select categorical and numerical columns
cat_cols = X.select_dtypes(include=["object", "string"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

print("\nCategorical Columns:")
print(list(cat_cols))

print("\nNumerical Columns:")
print(list(num_cols))

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

# Pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("model", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining model...")

# Train
model.fit(X_train, y_train)

print("Training completed!")

# Predict
pred = model.predict(X_test)

# Accuracy
score = r2_score(y_test, pred)
print(f"R2 Score: {score:.4f}")

# Save model
filename = "house_rent_prediction.pkl"

joblib.dump(model, filename)

print("\nModel saved successfully!")

# Verify file
if os.path.exists(filename):
    size = os.path.getsize(filename)
    print(f"File Size: {size} bytes")

    if size > 0:
        print("✅ Model file created successfully.")
    else:
        print("❌ Model file is empty!")
else:
    print("❌ Model file was not created.")