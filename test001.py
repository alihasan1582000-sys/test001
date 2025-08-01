import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
import numpy as np

# Load dataset
penguins = pd.read_csv("penguins.csv")

# Clean data
penguins = penguins[['species', 'body_mass_g', 'bill_length_mm']].dropna()

# Define features and target
X = penguins[['species', 'body_mass_g']]
y = penguins['bill_length_mm']

# Preprocessing: One-hot encode species
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), ['species'])
    ],
    remainder='passthrough'  # keep body_mass_g as-is
)

# Define pipeline
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Initial Test MSE: {mse:.2f}")

# Deployment function
def predict_beak_length(species, body_mass):
    input_df = pd.DataFrame({
        'species': [species],
        'body_mass_g': [body_mass]
    })
    pred = pipeline.predict(input_df)[0]
    return round(pred, 2)

# Function to retrain the model with new user data
def retrain_model_with_user_input(species, body_mass, actual_bill_length):
    global X_train, y_train, pipeline
    new_data = pd.DataFrame({
        'species': [species],
        'body_mass_g': [body_mass]
    })
    new_target = pd.Series([actual_bill_length])
    
    # Append new data
    X_train = pd.concat([X_train, new_data], ignore_index=True)
    y_train = pd.concat([y_train, new_target], ignore_index=True)

    # Retrain model
    pipeline.fit(X_train, y_train)
    print("Model updated with new user input.")

# 🔧 Example usage
print("\n--- PREDICTION ---")
species_input = "Adelie"
weight_input = 4000
predicted_beak = predict_beak_length(species_input, weight_input)
print(f"Predicted beak length: {predicted_beak} mm")

# 🎯 User corrects model by providing actual value
actual_beak_length = 41.0
retrain_model_with_user_input(species_input, weight_input, actual_beak_length)

# Try predicting again after retraining
print("\n--- AFTER RETRAINING ---")
new_prediction = predict_beak_length(species_input, weight_input)
print(f"New predicted beak length: {new_prediction} mm")
