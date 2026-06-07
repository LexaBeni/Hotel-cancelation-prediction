import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Hotel Booking Predictor",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Predictor")

page = st.sidebar.radio("Navigation", ['Overview', "Prediction", "Model Performance"])

if page == "Overview":
    st.header("Dataset Overview")

elif page == "Prediction":
    st.header("Prediction")

elif page == "Model Performance":
    st.header("Model Performance")