import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

st.set_page_config(
    page_title="Hotel Booking Predictor",
    layout="wide"
)

st.title("🏨 Hotel Booking Cancellation Predictor")

@st.cache_data
def load_dataset():
    return pd.read_csv(r'data\processed_hotel_booking.csv')
df = load_dataset()

page = st.sidebar.radio("Navigation", ['Overview', "Prediction", "Model Performance"])
selected_hotel = st.sidebar.selectbox("Hotel", ["All"] + list(df['hotel'].unique()))

if selected_hotel != "All":
    filtered_df = df[df["hotel"]==selected_hotel]
else:
    filtered_df = df.copy()

if page == "Overview":
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Reservations", len(filtered_df))
    with col2:
        st.metric("Cancellation Rate", f"{filtered_df['is_canceled'].mean()*100:.1f}%")
    with col3:
        st.metric("Average lead time", round(filtered_df['lead_time'].mean(), 2))
    cancel_df = filtered_df["is_canceled"].map({0: "Not canceled", 1:"Canceled"})
    cancel_df = cancel_df.value_counts()
    fig = px.pie(cancel_df, names=cancel_df.index, values=cancel_df.values, title='Reservation status distribution')
    st.plotly_chart(fig)
    hotel_counts = (df["hotel"].value_counts().reset_index())
    hotel_counts.columns = ["Hotel", "Count"]
    fig = px.bar(hotel_counts, x='Hotel', y='Count')
    st.plotly_chart(fig)

    fig = px.histogram(filtered_df, x='lead_time', color='is_canceled', nbins=40, title='Lead Time Distribution')
    st.plotly_chart(fig)

    deposit_counts = filtered_df.groupby('deposit_type')['is_canceled'].mean().reset_index()
    deposit_counts.columns = ['Type of deposit', "Is canceled"]
    fig = px.bar(deposit_counts, x='Type of deposit', y='Is canceled', title='Cancellation Rate by Deposit Type')
    st.plotly_chart(fig, use_container_width=True)


elif page == "Prediction":
    st.header("Prediction")

elif page == "Model Performance":
    st.header("Model Performance")

