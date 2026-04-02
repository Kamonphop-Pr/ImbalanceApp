import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('best_model.pkl')

st.title("Imbalanced Data Classification App")
st.write("This app uses a Logistic Regression model trained with SMOTE to predict the target class for the Mammography dataset.")

st.sidebar.header("Input Features")
# Mammography dataset has 6 features. Let's create input fields for them.
feature_0 = st.sidebar.number_input('Feature 0', value=0.0)
feature_1 = st.sidebar.number_input('Feature 1', value=0.0)
feature_2 = st.sidebar.number_input('Feature 2', value=0.0)
feature_3 = st.sidebar.number_input('Feature 3', value=0.0)
feature_4 = st.sidebar.number_input('Feature 4', value=0.0)
feature_5 = st.sidebar.number_input('Feature 5', value=0.0)

if st.button("Predict"):
    features = np.array([[feature_0, feature_1, feature_2, feature_3, feature_4, feature_5]])
    prediction = model.predict(features)
    prob = model.predict_proba(features)[0][1]

    st.write(f"### Prediction: {'Positive (1)' if prediction[0] == 1 else 'Negative (0)'}")
    st.write(f"### Probability of Positive Class: {prob:.4f}")
