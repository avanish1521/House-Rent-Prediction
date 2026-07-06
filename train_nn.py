import pandas as pd
import joblib
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# ==========================
# Load Dataset
# ==========================
df = pd.read_csv("House_Rent_Dataset.csv")

print("Dataset Loaded Successfully!\n")
print(df.head())

# ==========================
# Drop Unnecessary Columns
# ==========================
drop_cols = [
    "Posted On",
    "Point of Contact",
    "Area Locality"
]

for col in drop_cols:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# ==========================
# Features and Target
# ==========================
X = df.drop("Rent", axis=1)
y = df["Rent"]

# ==========================
# Find Categorical & Numerical Columns
# ==========================
cat_cols = X.select_dtypes(include=["object", "string"]).columns
num_cols = X.select_dtypes(include=["int64", "float64"]).columns

print("\nCategorical Columns:")
print(cat_cols)

print("\nNumerical Columns:")
print(num_cols)

# ==========================
# Preprocessing
# ==========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

X_processed = preprocessor.fit_transform(X)

if hasattr(X_processed, "toarray"):
    X_processed = X_processed.toarray()

print("\nShape After Preprocessing:", X_processed.shape)

# ==========================
# Train Test Split
# ==========================
X_train, X_test, y_train, y_test = train_test_split(
    X_processed,
    y,
    test_size=0.20,
    random_state=42
)

print("Training Data :", X_train.shape)
print("Testing Data  :", X_test.shape)

# ==========================
# Build Neural Network
# ==========================
model = Sequential([
    Dense(128, activation="relu", input_shape=(X_train.shape[1],)),
    Dense(64, activation="relu"),
    Dense(32, activation="relu"),
    Dense(1)
])

model.summary()

# ==========================
# Compile Model
# ==========================
model.compile(
    optimizer="adam",
    loss="mean_squared_error",
    metrics=["mae"]
)

# ==========================
# Early Stopping
# ==========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=10,
    restore_best_weights=True
)

# ==========================
# Train Model
# ==========================
history = model.fit(
    X_train,
    y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=32,
    callbacks=[early_stop],
    verbose=1
)

# ==========================
# Evaluate Model
# ==========================
loss, mae = model.evaluate(X_test, y_test, verbose=0)

pred = model.predict(X_test)

# ==========================
# Metrics
# ==========================
r2 = r2_score(y_test, pred)

mae_score = mean_absolute_error(y_test, pred)

rmse = np.sqrt(mean_squared_error(y_test, pred))

print("\n==============================")
print("Neural Network Performance")
print("==============================")

print(f"R² Score : {r2:.4f}")
print(f"MAE      : {mae_score:.2f}")
print(f"RMSE     : {rmse:.2f}")
print(f"Loss     : {loss:.2f}")

# ==========================
# Sample Prediction
# ==========================
print("\nSample Predictions")

sample_prediction = model.predict(X_test[:5])

for i in range(5):
    print(f"Actual : {y_test.iloc[i]:,.0f}   Predicted : {sample_prediction[i][0]:,.0f}")

# ==========================
# Save Model
# ==========================
model.save("house_rent_nn.keras")

joblib.dump(preprocessor, "preprocessor.pkl")

print("\n✅ Neural Network Saved Successfully!")
print("✅ Preprocessor Saved Successfully!")