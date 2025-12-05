import streamlit as st
import pandas as pd
import pickle

@st.cache_resource
def load_pipeline():
    with open("files/full_model_pipeline.pkl", "rb") as f:
        return pickle.load(f)
    
def group_owner_category(owner: str) -> str:
    if pd.isna(owner):
        return "Unknown"
    if owner in {"Third Owner", "Fourth & Above Owner", "Test Drive Car"}:
        return "Other Owner"
    return owner
    
pipeline = load_pipeline()

# === Streamlit UI ===
st.title("Car Price Predictor")

name = st.text_input("Car name", "Maruti Swift")
year = st.number_input("Year", min_value=1990, max_value=2025, value=2018)
km_driven = st.number_input("Kilometers driven", min_value=0, value=35000)
fuel = st.selectbox("Fuel type", ["Petrol", "Diesel", "CNG", "LPG"])
seller_type = st.selectbox("Seller type", ["Individual", "Dealer", "Trustmark Dealer"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
owner = st.selectbox("Owner", ["First Owner", "Second Owner", "Third Owner", "Fourth & Above Owner", "Test Drive Car"])
engine = st.number_input("Engine (cc)", min_value=500, max_value=5000, value=1200)
max_power = st.number_input("Max power (bhp)", min_value=30, max_value=500, value=80)
seats = st.number_input("Seats", min_value=2, max_value=10, value=5)
mileage = st.number_input("Mileage", min_value=10, max_value=100, value=25)

if st.button("Predict Price"):
    user_input = pd.DataFrame([{
        'name': name,
        'year': int(year),
        'km_driven': int(km_driven),
        'fuel': fuel,
        'seller_type': seller_type,
        'transmission': transmission,
        'owner': owner,
        'engine': float(engine),
        'max_power': float(max_power),
        'seats': str(seats),
        'mileage': int(mileage)
    }])
    
    user_input["owner"] = user_input["owner"].apply(group_owner_category)

    try:
        predicted_price = pipeline.predict(user_input)[0]
        st.success(f"Predicted Selling Price: **{predicted_price:,.0f}**")
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        st.write("Check that all feature names and types match the training data.")