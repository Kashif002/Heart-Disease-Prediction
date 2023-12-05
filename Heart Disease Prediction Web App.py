# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 00:32:00 2023

@author: kashi
"""

import numpy as np
import pickle
import json
import requests
import streamlit as st
from streamlit_lottie import st_lottie
loaded_model=pickle.load(open('C:/Kashif/Kashif College/Lecture Notes/SEM-7/Project/Prediction Model/trained_model.sav','rb'))

#Giving Animations
def load_lottieurl(url:str):
    r=requests.get(url)
    if r.status_code!=200:
        return None
    return r.json()

#Creating a function for prediction

def heart_prediction(input_data):
    
    input_data=[float(value) for value in input_data[0]]
    #Change the input  data to a numpy array
    input_data_as_numpy_array=np.asarray(input_data)

    #Reshape the numpy array as we are predicting for only one instance
    input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

    prediction=loaded_model.predict(input_data_reshape)

    if(prediction[0]==0):
      return "The Person does not have Heart Disease"
    else:
      return "The Person has Heart Disease"

    
def main():
    
    #Giving a title
    st.title('Heart Disease Prediction Web App')
    
    lottie_hello=load_lottieurl("https://lottie.host/ae3af931-11d5-4c44-8124-56196cb25513/qFC5voY370.json")
    st_lottie(lottie_hello,key="hello")
    
    #Getting the input data from the user
    #colums for input fields
    col1,col2,col3=st.columns(3)
    
    with col1:
        age=st.text_input("Age")
    with col2:
        sex=st.text_input("Sex")
    with col3:
        cp=st.text_input("Chest Pain Type")
    with col1:
        trestbps=st.text_input("Resting Blood Pressure")
    with col2:
        chol=st.text_input("Serum Cholestoral in mg/d")
    with col3:
        fbs=st.text_input("Fasting Blood Sugar > 120 mg/dl")
    with col1:
        restecg=st.text_input("Resting Electrocardiographic Results")
    with col2:
        thalach=st.text_input("Maximum Heart Rate Achieved")
    with col3:
        exang=st.text_input("Exercise Induced Angina")
    with col1:
        oldpeak=st.text_input("ST depression induced by exercise")
    with col2:
        slope=st.text_input("Slope of the peak exercise ST segment")
    with col3:
        ca=st.text_input("Major vessels colored by flourosopy")
    with col1:
        thal=st.text_input("Thal: 0 = Normal; 1 = Fixed defect; 2 = Reversable defect")
    
    
    
    #code for prediction
    diagnosis= ''
    
    #Creating a button for prediction
    if st.button("Heart Disease Test Result"):
        diagnosis= heart_prediction([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
        
    st.success(diagnosis)
    

if __name__ =='__main__':
    main()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    