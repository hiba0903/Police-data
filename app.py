import streamlit as st


import folium
from streamlit_folium import st_folium

import pandas as pd
import pickle
from datetime import datetime
from geopy.geocoders import Nominatim

# Load the pre-trained model and columns
with open("voting_model.pkl", "rb") as file:
    voting_model = pickle.load(file)

with open("columns.pkl", "rb") as file:
    model_columns = pickle.load(file)

# Function to get current date and day
def get_current_date():
    now = datetime.now()
    current_day = now.strftime('%A')  # Get the weekday name
    current_month = now.strftime('%B')  # Get the full month name
    current_date = now.strftime('%d')  # Get the current day of the month
    return current_month, current_day, current_date, now

# Get current date and day
current_month, current_day, current_date, current_datetime = get_current_date()

# Set the title and subtitle
st.title("Accident Prediction Model")
st.write("This model predicts the likelihood of accidents based on weather, visibility, and road conditions.")

# Date input section: Use date_input to let the user select a date
selected_date = st.date_input("Select a Date", current_datetime)

# Extract the month and day from the selected date
selected_month = selected_date.month
selected_day_of_week = selected_date.weekday() + 1  # Convert to 1=Monday, 7=Sunday

# Display the selected date in a user-friendly format
st.write(f"### Current Date: {selected_date.strftime('%A')}, {selected_date.day} {selected_date.strftime('%B')}")

# Location selection via map
st.subheader("Select Location on the Map:")

# Center the map at a default location (e.g., India) or any preferred location
map_center = [20.5937, 78.9629]  # Center on India
m = folium.Map(location=map_center, zoom_start=5)

# Display the map using Streamlit
clicked_location = st_folium(m, width=700)

# Initialize Geolocator
geolocator = Nominatim(user_agent="accident_prediction_app")

# Function to get place name based on latitude and longitude
def get_place_name(lat, lon):
    try:
        location = geolocator.reverse((lat, lon), language="en", timeout=10)
        if location:
            return location.address
        else:
            return "Unknown location"
    except Exception as e:
        return f"Error retrieving location: {e}"

# Check if a location has been clicked
latitude, longitude = None, None
if clicked_location and 'last_clicked' in clicked_location and clicked_location['last_clicked']:
    latitude = clicked_location['last_clicked']['lat']
    longitude = clicked_location['last_clicked']['lng']
    # Get the place name from the latitude and longitude
    place_name = get_place_name(latitude, longitude)
    st.write(f"### Selected Location: {place_name}")
else:
    latitude, longitude = map_center
    st.write(f"### Default Location Selected: {map_center}")

# Input fields for user data (weather, area type, visibility, etc.)
weather = st.selectbox("Weather", ["Sunny/Clear", "Mist/Fog", "Dust Storm", "Heavy rain", "Light rain", "Snow", "Very Cold", "Very Hot"])
area_type = st.selectbox("Type of Area", ["Urban", "Rural"])
visibility = st.selectbox("Visibility", ["Good", "Poor", "Not Known"])
road_type = st.selectbox("Type of Road", ["National Highway", "State Highway", "MDR", "ODR", "Other Road"])
road_features = st.selectbox("Road Features", ["Straight Road", "Curved Road", "Culvert", "Ongoing Road Works/Under Construction", "Others", "Pot Holes", "Steep Grade"])

# Create the input data for prediction
input_data = {
    "Weather": [weather],
    "Type Area": [area_type],
    "Visibility": [visibility],
    "Type Road": [road_type],
    "Road Features": [road_features],
    "Month": [selected_month],
    "Day_of_Week": [selected_day_of_week],
    "Latitude": [latitude],
    "Longitude": [longitude]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# One-hot encode the input data using the same columns as the training data
input_encoded = pd.get_dummies(input_df, drop_first=True)

# Ensure that the input features match the training model
input_encoded = input_encoded.reindex(columns=model_columns, fill_value=0)

# Predict button
if st.button('Predict Accident Occurrence'):
    # Make the prediction
    y_prob_sample = voting_model.predict_proba(input_encoded)[:, 1][0]  # Get probability for class '1' (accident occurred)
    result = "Yes" if y_prob_sample >= 0.5 else "No"
    
    # Display result
    st.write(f"### Predicted Accident Occurrence: **{result}**")
