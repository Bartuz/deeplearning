# Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# Fix Error #15 - libiomp5.dylib already initialized
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

# 1: Data preprocessing

# Importing the dataset
dataset = pd.read_csv('churn_bank.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data (independent variables)
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
# Fix dummy variable trap
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# 2: Build ANN

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()
# first hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))
# second hidden layer
classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
# output layer
classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size=10, epochs=100)
# 3: Making prediction and evaluating the model
