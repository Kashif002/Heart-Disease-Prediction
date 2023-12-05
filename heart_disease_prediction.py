# -*- coding: utf-8 -*-
"""Heart Disease Prediction.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/181BLUJ1xAzZIEDNjExACrA7hdZJyDWh6

**Importing the Dependencies**
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from matplotlib import rcParams
import warnings
warnings.filterwarnings('ignore')

"""**Data Collection and Processing**"""

#Loading the csv data to a Pandas DataFrame

 heart_data =pd.read_csv('/content/data.csv')

#Print first 5 rows of the dataset
 heart_data.head()

#print last 45 rows of the dataset
heart_data.tail()

#The number of rows and columns in the dataset
 heart_data.shape

#Getting some info about the data
 heart_data.info()

#Checking for missing values
heart_data.isnull().sum()

#Statsitical measures about the data
heart_data.describe()

#Checking the distribution of target variable
heart_data['target'].value_counts()

"""1 --> Defective Heart

0 --> Healthy Heart

**Data Visualization**
"""

plt.matshow(heart_data.corr())
plt.yticks(np.arange(heart_data.shape[1]), heart_data.columns)
plt.xticks(np.arange(heart_data.shape[1]), heart_data.columns)
plt.colorbar()

heart_data.hist()

plt.bar(heart_data['target'].unique(), heart_data['target'].value_counts(), color = ['red', 'green'])
plt.xticks([0, 1])
plt.xlabel('Target Classes')
plt.ylabel('Count')
plt.title('Count of each Target Class')

"""**Splitting the Features and Target**"""

X=heart_data.drop(columns='target',axis=1) #having only feature data in X
 Y=heart_data['target'] #having only target data in Y

print(X)

print(Y)

"""**Splitting the Data into Training Data and Test Data**"""

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,X_train.shape,X_test.shape)

"""**Model Training**

**Logistic Regression**
"""

model=LogisticRegression()

#Training the LogisticRegression Model with Training Data
model.fit(X_train,Y_train)

"""**Decision Tree**"""

modelD = DecisionTreeClassifier()

modelD.fit(X_train,Y_train)

"""**Model Evaluation**

**Accuracy Score**
"""

#Accuracy of Logistic Regression Models for Training Data
X_train_prediction=model.predict(X_train)
training_data_accuracy= accuracy_score(X_train_prediction, Y_train)

#Accuracy of Decision Tree Models for Training Data
X_train_prediction=modelD.predict(X_train)
training_data_accuracyD= accuracy_score(X_train_prediction, Y_train)

print("Accuracy of Logistic Regression Models for Training Data: ",training_data_accuracy)
print("Accuracy of Decisison Tree Models for Training Data: ",training_data_accuracyD)

#Accuracy of Logistic Regression Model for Test Data
X_test_prediction=model.predict(X_test)
test_data_accuracy= accuracy_score(X_test_prediction, Y_test)

#Accuracy of Decision Tree Model for Test Data
X_test_prediction=modelD.predict(X_test)
test_data_accuracyDT= accuracy_score(X_test_prediction, Y_test)

print("Accuracy of Logistic Regression for Test Data : ",test_data_accuracy)
print("Accuracy of Decision Tree for Test Data : ",test_data_accuracyDT)

"""**Prediction Model**"""

input_data=(57,1,0,140,192,0,1,148,0,0.4,1,0,1)

#Change the input  data to a numpy array
input_data_as_numpy_array=np.asarray(input_data)

#Reshape the numpy array as we are predicting for only one instance
input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

prediction=model.predict(input_data_reshape)

if(prediction[0]==0):
  print("The Person does not have Heart Disease")
else:
  print("The Person has Heart Disease")

import pickle

filename='trained_model.sav'
pickle.dump(model,open(filename,'wb'))

#Loading the saved model
loaded_model=pickle.load(open('trained_model.sav','rb'))

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