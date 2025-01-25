# -*- coding: utf-8 -*-
"""
Created on Fri Jan 24 19:48:40 2025

@author: HP-PC
"""

import numpy as np
import pickle
import streamlit as st  # Corrected the import statement

# Load the trained model
loaded_model = pickle.load(open("C:/Users/HP-PC/Desktop/Python data analytics/Machine Learning model/trained_model.sav", 'rb'))

# Function for diabetes prediction
def diabetes_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)  # Fixed missing parenthesis
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# Main function for Streamlit UI
def main():
    st.title('Diabetes Prediction Web App')  # Fixed missing parenthesis
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure value')
    SkinThickness = st.text_input('Skin Thickness value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
    Age = st.text_input('Age of the Person')
    
    diagnosis = ''
    if st.button('Diabetes Test Result'):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        st.success(diagnosis)

# Run the app
if __name__ == '__main__':
    main()













