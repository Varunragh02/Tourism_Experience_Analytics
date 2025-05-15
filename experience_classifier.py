# experience_classifier.py

import streamlit as st
import pandas as pd
import joblib
import numpy as np

# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Tourism Experience Classification", layout="centered")

# --- Load Model and Encoders ---
@st.cache_resource
def load_model_and_encoders():
    model = joblib.load("D:/Guvi/Tourism_Experience_Analytics/best_classification_model.pkl")
    label_encoders = joblib.load("D:/Guvi/Tourism_Experience_Analytics/label_encoders.pkl")
    location_df = pd.read_csv("D:/Guvi/Tourism_Experience_Analytics/location_mapping.csv")

    # Clean placeholder values
    location_df = location_df[
        (location_df['Continent'] != '-') &
        (location_df['Country'] != '-') &
        (location_df['Region'] != '-') &
        (location_df['CityName'] != '-')
    ]

    return model, label_encoders, location_df

model, label_encoders, location_df = load_model_and_encoders()

# --- UI Setup ---
st.title("🌍 Tourism Experience Classification")
st.markdown("Predict the visitor **experience category** based on destination, region, attraction type, and date.")

# --- User Inputs ---
continents = location_df['Continent'].unique()
attraction_types = [
    "Nature & Wildlife Areas", "Beaches", "Religious Sites", "Water Parks",
    "Points of Interest & Landmarks", "Volcanos", "Waterfalls", "Flea & Street Markets",
    "History Museums", "Historic Sites", "Ancient Ruins", "National Parks",
    "Ballets", "Caverns & Caves", "Neighborhoods", "Speciality Museums", "Spas"
]

continent = st.selectbox("Continent", continents)
attraction_type = st.selectbox("Attraction Type", attraction_types)

filtered_countries = location_df[location_df['Continent'] == continent]['Country'].unique()
country = st.selectbox("Country", filtered_countries)

filtered_regions = location_df[location_df['Country'] == country]['Region'].unique()
region = st.selectbox("Region", filtered_regions)

filtered_cities = location_df[location_df['Region'] == region]['CityName'].unique()
city = st.selectbox("City", filtered_cities)

visit_month = st.selectbox("Visit Month", list(range(1, 13)), format_func=lambda x: pd.to_datetime(str(x), format="%m").strftime("%B"))
visit_year = st.selectbox("Visit Year", list(range(2015, 2026)))

# --- Safe Encoding Function ---
def safe_label_encode(encoder, value, field_name):
    try:
        return encoder.transform([value])[0]
    except ValueError:
        st.warning(f"⚠️ The {field_name} value '{value}' was not seen during training. Please choose another.")
        st.stop()

# --- Encode Inputs ---
region_encoded = safe_label_encode(label_encoders['Region'], region, "Region")
city_encoded = safe_label_encode(label_encoders['CityName'], city, "City")
attraction_encoded = safe_label_encode(label_encoders['AttractionType'], attraction_type, "Attraction Type")

# --- Construct Input DataFrame ---
df_input = pd.DataFrame({
    'Region': [region_encoded],
    'CityName': [city_encoded],
    'AttractionType': [attraction_encoded],
    'VisitMonth': [visit_month],
    'VisitYear': [visit_year]
})

# One-hot encode Continent and Country
categorical_flags = {
    f'Continent_{continent}': 1,
    f'Country_{country}': 1,
    'AttractionTypeId': attraction_encoded
}

# Add dummy columns expected by model
dummy_cols = [col for col in model.feature_names_in_ if col.startswith('Continent_') or col.startswith('Country_')]
for col in dummy_cols:
    df_input[col] = categorical_flags.get(col, 0)

# Ensure all required model input columns are present
for col in model.feature_names_in_:
    if col not in df_input.columns:
        df_input[col] = 0

# Reorder columns to match model input
df_input = df_input[model.feature_names_in_]

# --- Class Mapping ---
class_mapping = {
    0: 'Couples',
    1: 'Friends',
    2: 'Family',
    3: 'Solo',
    4: 'Business'
}

# --- Predict ---
if st.button("Classify Experience"):
    prediction = model.predict(df_input)[0]
    st.write("🔍 Raw model prediction:", prediction)

    # Handle numeric or string class prediction
    if isinstance(prediction, (int, float, np.integer)):
        class_label = class_mapping.get(int(prediction), "Unknown")
    else:
        class_label = str(prediction)

    st.success(f"⭐ Predicted Experience Category: **{class_label}**")
    st.balloons()
