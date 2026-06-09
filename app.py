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
@st.cache_resource
def load_model():
    return joblib.load("models/xgb_model.joblib")
model = load_model()
sample = df.drop("is_canceled", axis=1).iloc[[0]].copy()

page = st.sidebar.radio("Navigation", ['Overview', "Prediction", "Model Performance"])
selected_hotel = st.sidebar.selectbox("Hotel", ["All"] + list(df['hotel'].unique()))

if selected_hotel != "All":
    filtered_df = df[df["hotel"]==selected_hotel]
else:
    filtered_df = df.copy()

if page == "Overview":
    st.markdown("""
        ## Project Overview

        This machine learning project predicts whether a hotel reservation
        will be cancelled.

        The model was trained on over 119,000 reservations and achieved:

        - ROC-AUC: 95%
        - Accuracy: 88%
        - F1-score: 84%

        The application allows users to:

        - Explore the dataset
        - Predict cancellation probability
        - Review model performance
        """)
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

    with st.form("Prediction form"):
        hotel = st.selectbox("Hotel", df['hotel'].unique())
        lead = st.slider("Lead time", 0, 750, 100)
        adults = st.number_input("Number of Adults", 1, 10, 2)
        children = st.number_input("Number of children", 0, 50, 0)
        deposit = st.selectbox("Deposit type", df['deposit_type'].unique())
        market_segment = st.selectbox("Market segment", df['market_segment'].unique())
        adr = st.slider("Adr", 0, 5000, 100)
        is_repeated_guest = st.checkbox("Is repeated guest?")
        total_of_special_requests = st.number_input("Number of special guests", 0, 10, 0)

        submitted = st.form_submit_button("Predict")


        if submitted:
            input_df = pd.DataFrame({
                "hotel": [hotel],
                "lead_time": [lead],
                "adults":[adults],
                "children": [children],
                "deposit_type": [deposit],
                "market_segment" : [market_segment],
                "total_guests": [adults  + children],
                "adr": [adr],
                "is_repeated_guest": [int(is_repeated_guest)],
                "total_of_special_requests" : [total_of_special_requests]
            })
            st.table(input_df)
            sample["hotel"] = hotel
            sample["lead_time"] = lead
            sample["adults"] = adults
            sample["children"] = children
            sample["deposit_type"] = deposit
            sample["market_segment"] = market_segment
            sample["total_guests"] = adults + children
            sample["adr"] = adr
            if is_repeated_guest:
                sample["is_repeated_guest"] = 1
            else:
                sample["is_repeated_guest"] = 0
            sample["total_of_special_requests"] = total_of_special_requests
            prob = model.predict_proba(sample)[0, 1]
            st.metric("Cancellation Probability", f"{prob:.1%}")
            st.progress(float(prob))
            if prob < 0.3:
                st.success("Low cancellation risk")

            elif prob < 0.7:
                st.warning("Medium cancellation risk")

            else:
                st.error("High cancellation risk")
elif page == "Model Performance":
    st.header("Model Performance")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Accuracy", "88.0%")

    with col2:
        st.metric("ROC-AUC", "95.1%")

    with col3:
        st.metric("F1 Score", "84.0%")
    
    st.subheader("Confusion Matrix")
    st.image("images/confusion_matrix.png")
    st.subheader("Feature Importance")
    st.image("images/feature_importance.png")
    st.subheader("SHAP Summary")
    st.image("images/shap_summary.png")


