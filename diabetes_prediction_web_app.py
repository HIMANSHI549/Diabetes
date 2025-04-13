# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 15:44:39 2025

@author: Admin1
"""

import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open('C:/Users/Admin1/Downloads/ml/trained_model.sav','rb'))
#creating function

def diabetes_prediction(input_data):
    

    # changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)



    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
      return 'The person is not diabetic'
    else:
      return 'The person is diabetic'


def main():
    #giving title
    st.title('Diabetes Prediction web app')
    
    Pregnancies=st.text_input('Number of Pregnancies')
    Glucose=st.text_input('Glucose level')
    BloodPressure=st.text_input('BP value')
    SkinThickness=st.text_input('SkinThickness value')
    Insulin=st.text_input('Insulin level')
    BMI=st.text_input('BMI value')
    DiabetesPedigreeFunction=st.text_input('Diabetes pedigree value')
    Age=st.text_input('Age of person')
    
    diagnosis=''
    
    #creating button
    if st.button('Diabetes test result'):
        diagnosis=diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
    
    st.success(diagnosis)
    
    
if __name__ == '__main__':
    main()
    
    