import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("penguins.csv")
    df = df[['species', 'body_mass_g', 'bill_length_mm']].dropna()
    return df

data = load_data()

# Define features and target
X = data[['species', 'body_mass_g']]
y = data['bill_length_mm']

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('species', OneHotEncoder(handle_unknown='ignore'), ['species'])
], remainder='passthrough')

pipeline = Pipeline([
    ('preprocess', preprocessor),
    ('regressor', LinearRegression())
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
pipeline.fit(X_train, y_train)

# Streamlit app UI
st.title("🐧 Penguin Beak Length Predictor")
st.markdown("Predict the beak length of a penguin using its species and weight. Provide real values to help the model learn.")

# Input fields
species_input = st.selectbox("Select species", options=data['species'].unique())
body_mass_input = st.number_input("Enter body mass (g)", min_value=2500, max_value=7000, step=50)

if st.button("Predict Beak Length"):
    input_df = pd.DataFrame({
        'species': [species_input],
        'body_mass_g': [body_mass_input]
    })
    prediction = pipeline.predict(input_df)[0]
    st.success(f"Predicted beak length: {round(prediction, 2)} mm")

# Optional retraining section
st.markdown("---")
st.header("🔁 Improve the Model with Real Data")
st.markdown("If you know the actual beak length, submit it to improve predictions.")
actual_beak = st.number_input("Actual beak length (mm)", min_value=30.0, max_value=70.0, step=0.1)

if st.button("Update Model"):
    new_data = pd.DataFrame({
        'species': [species_input],
        'body_mass_g': [body_mass_input]
    })
    new_target = pd.Series([actual_beak])
    X_train = pd.concat([X_train, new_data], ignore_index=True)
    y_train = pd.concat([y_train, new_target], ignore_index=True)
    pipeline.fit(X_train, y_train)
    st.success("✅ Model updated with new data!")
    new_prediction = pipeline.predict(new_data)[0]
    st.info(f"New predicted beak length: {round(new_prediction, 2)} mm")
