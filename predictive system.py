# -*- coding: utf-8 -*-
"""
Created on Sun Sep 17 00:24:01 2023

@author: kashi
"""

import numpy as np
import pickle
loaded_model=pickle.load(open('C:/Kashif/Kashif College/Lecture Notes/SEM-7/Project/Prediction Model/trained_model.sav','rb'))

input_data=(57,1,0,140,192,0,1,148,0,0.4,1,0,1)

#Change the input  data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#Reshape the numpy array as we are predicting for only one instance
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

prediction=loaded_model.predict(input_data_reshape)

if(prediction[0]==0):
  print("The Person does not have Heart Disease")
else:
  print("The Person has Heart Disease")
