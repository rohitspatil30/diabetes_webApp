import numpy as np
import pandas as pd
import pickle
import streamlit as st

# Load the model (ensure this is done before calling the function)
with open('D:/languages/python/machine_learning/Notes/diabetes_ML_WebApp/trained_model.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

def heart_disease_prediction(input_data):
    """
    Predict whether a person has heart disease based on input features.
    
    Parameters:
    input_data (list or array-like): List or array-like object containing feature values.
    
    Returns:
    str: Prediction result.
    """
    
    # Ensure input_data is a list or array-like
    if not isinstance(input_data, (list, np.ndarray)):
        raise ValueError("Input data should be a list or a numpy array")

    # Convert the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Check that the number of features matches the model's expected input
    if input_data_as_numpy_array.shape[0] != 13:
        raise ValueError("Input data must have 13 features")

    # Reshape the array to be 2D (1 instance, 13 features)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    # Make a prediction
    prediction = loaded_model.predict(input_data_reshaped)

    # Interpret the prediction
    if prediction[0] == 0:
        return 'The person is not having heart disease'
    else:
        return 'The person is having heart disease'

def main():
    # Giving a title
    st.title('Heart Disease Prediction Web App')

    # Text input fields
    age = st.text_input('What is your age')
    sex = st.text_input('What is your sex')
    chest_pain_type = st.text_input('Enter type of chest pain')
    resting_bp = st.text_input('What is your resting Blood pressure')
    cholestoral = st.text_input('What is your Cholestrol')
    fasting_blood_sugar = st.text_input('fasting_blood_sugar')
    restecg = st.text_input('restecg')
    max_hr = st.text_input('max_hr')
    exang = st.text_input('exang')
    oldpeak = st.text_input('oldpeak')
    slope = st.text_input('slope')
    num_major_vessels = st.text_input('num_major_vessels')
    thal = st.text_input('thal')

    # Convert inputs to floats
    try:
        input_data = [
            float(age),
            float(sex),
            float(chest_pain_type),
            float(resting_bp),
            float(cholestoral),
            float(fasting_blood_sugar),
            float(restecg),
            float(max_hr),
            float(exang),
            float(oldpeak),
            float(slope),
            float(num_major_vessels),
            float(thal)
        ]
    except ValueError:
        st.error("Please enter valid numeric values.")
        return

    # Code for Prediction
    diagnosis = ''
    
    # Creating a button for Prediction
    if st.button('Heart Disease Test Result'):
        try:
            diagnosis = heart_disease_prediction(input_data)
            st.success(diagnosis)
        except ValueError as e:
            st.error(f"Error: {e}")

if __name__ == '__main__':
    main()
