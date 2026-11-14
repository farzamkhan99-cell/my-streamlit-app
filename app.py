import streamlit as st
import pandas as pd
import pickle
from pathlib import Path

# Load the trained model and the feature input schema
BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / 'pipeline.pkl', 'rb') as file:
    model = pickle.load(file)

with open(BASE_DIR / 'my_feature_dict.pkl', 'rb') as file:
    my_feature_dict = pickle.load(file)

# Function to predict Employee Churn
def predict_churn(data):
    prediction = model.predict(data)
    return prediction

st.title('Employee Churn Prediction App')
st.subheader('Created by Muhammad Farzam Khan')
st.write('Enter employee details to predict whether an employee will leave the company.')

with st.form('employee_churn_form'):
    # Categorical Features from exported schema
    st.subheader('Categorical Features')
    categorical_input = my_feature_dict.get('CATEGORICAL')
    categorical_input_vals = {}

    for i, col in categorical_input.get('Column Name').items():
        categorical_input_vals[col] = st.selectbox(
            col,
            categorical_input.get('Members')[i],
            key=col
        )

    # Numerical Features from exported schema
    st.subheader('Numerical Features')
    numerical_input = my_feature_dict.get('NUMERICAL')
    numerical_input_vals = {}

    for col in numerical_input.get('Column Name'):
        numerical_input_vals[col] = st.number_input(
            col,
            min_value=int(numerical_input.get('Min')[col]),
            max_value=int(numerical_input.get('Max')[col]),
            value=int(numerical_input.get('Default')[col]),
            step=1,
            key=col
        )

    # Interaction values will be passed to the fitted model
    submitted = st.form_submit_button('Predict Employee Churn')

if submitted:
    input_values = {**categorical_input_vals, **numerical_input_vals}
    prediction_data = pd.DataFrame([input_values])
    prediction = predict_churn(prediction_data)[0]

    st.subheader('Model Prediction')
    if prediction == 1:
        st.error('Prediction: Employee is likely to LEAVE the company.')
    else:
        st.success('Prediction: Employee is likely to STAY with the company.')

    # Bonus: show class-1 probability as reported by the fitted model.
    leave_probability = model.predict_proba(prediction_data)[0][1]
    st.write(f'Estimated probability of leaving: {leave_probability:.1%}')
