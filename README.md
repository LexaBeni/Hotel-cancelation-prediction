# Hotel Booking Cancellation Predictor

## Overview

This machine learning project predicts whether a hotel reservation will be cancelled.

The model was trained on the Hotel Booking Demand dataset containing over 119,000 reservations.

## Project Workflow

1. Data Analysis
2. Feature Engineering
3. Model Training
4. Hyperparameter Tuning
5. Model Interpretation using SHAP
6. Streamlit Application

## Models Tested

* XGBoost
* HistGradientBoosting

## Best Results

| Metric   | Score |
| -------- | ----- |
| Accuracy | 88.0% |
| ROC-AUC  | 95.1% |
| F1 Score | 84.0% |

## Technologies

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* SHAP
* Plotly
* Streamlit

## Project Structure

```text
hotel-booking-project/
├── data/
├── notebooks/
├── models/
├── images/
├── app.py
├── requirements.txt
└── README.md
```
## Live Demo
https://hotel-cancelation-prediction-zecfdq4n7hdgkfahqw58nj.streamlit.app/

## Run Application

```bash
streamlit run app.py
```
