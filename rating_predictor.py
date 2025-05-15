import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model and encoders
model = joblib.load("D:/Guvi/Tourism_Experience_Analytics/best_model.pkl")
label_encoders = joblib.load("D:/Guvi/Tourism_Experience_Analytics/label_encoders.pkl")
scaler = joblib.load("D:/Guvi/Tourism_Experience_Analytics/scaler.pkl")

# Load the location mapping CSV
location_df = pd.read_csv("D:/Guvi/Tourism_Experience_Analytics/location_mapping.csv")

# Extract unique values for continent, country
continents = location_df['Continent'].unique()
countries = location_df['Country'].unique()

# Define AttractionType options
attraction_types = [
    "Nature & Wildlife Areas", "Beaches", "Religious Sites", "Water Parks",
    "Points of Interest & Landmarks", "Volcanos", "Waterfalls", "Flea & Street Markets",
    "History Museums", "Historic Sites", "Ancient Ruins", "National Parks",
    "Ballets", "Caverns & Caves", "Neighborhoods", "Speciality Museums", "Spas"
]

# --- Streamlit UI ---
st.set_page_config(page_title="Tourism Rating Predictor", layout="centered")
st.title("🌍 Tourism Rating Predictor")
st.markdown("Predict user experience rating for a tourist attraction based on various factors.")

# --- Collect Inputs ---
visit_mode = st.selectbox("Visit Mode", ['Group', 'Solo', 'Family', 'Couple'])
continent = st.selectbox("Continent", continents)
attraction_type = st.selectbox("Attraction Type", attraction_types)

# Filter countries by continent
filtered_countries = location_df[location_df['Continent'] == continent]['Country'].unique()
country = st.selectbox("Country", filtered_countries)

# Filter regions and cities
filtered_regions = location_df[location_df['Country'] == country]['Region'].unique()
region = st.selectbox("Region", filtered_regions)

filtered_cities = location_df[location_df['Region'] == region]['CityName'].unique()
city = st.selectbox("City", filtered_cities)

# Simulate User Aggregates
avg_rating = st.slider("User's Avg Rating", 1.0, 5.0, 4.0, step=0.1)
total_visits = st.slider("User's Total Visits", 1, 100, 10)

# --- Encoding ---
region_encoded = label_encoders['Region'].transform([region])[0]
city_encoded = label_encoders['CityName'].transform([city])[0]
attraction_encoded = label_encoders['AttractionType'].transform([attraction_type])[0]

# --- DataFrame Setup ---
df_input = pd.DataFrame({
    'Region': [region_encoded],
    'CityName': [city_encoded],
    'AttractionType': [attraction_encoded]
})

# One-hot encode VisitMode, Continent, Country
categorical_cols = {
    'VisitMode_' + visit_mode: 1,
    'Continent_' + continent: 1,
    'Country_' + country: 1,
    'AttractionTypeId': attraction_encoded
}

# Add all dummy columns required by model
dummy_cols = []
for cat in ['VisitMode', 'Continent', 'Country']:
    dummy_cols += [col for col in model.feature_names_in_ if col.startswith(cat + "_")]

for col in dummy_cols:
    df_input[col] = categorical_cols.get(col, 0)

# Scale aggregates
normalized = scaler.transform([[avg_rating, total_visits]])
df_input['Normalized_Rating'] = normalized[0][0]
df_input['Normalized_Visits'] = normalized[0][1]

# Add any missing columns
for col in model.feature_names_in_:
    if col not in df_input:
        df_input[col] = 0

# Reorder to match model input
df_input = df_input[model.feature_names_in_]

# --- Prediction ---
if st.button("Predict Rating"):
    prediction = model.predict(df_input)[0]
    st.success(f"⭐ Predicted Rating: **{prediction:.2f}**")
    st.balloons()