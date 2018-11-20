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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

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

classifier.fit(X_train, y_train, batch_size=10, epochs=10)
# 3: Making prediction and evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Bonus: Prediction for single customer
ab = sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]]))
new_prediction = classifier.predict(sc.transform(np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
prediction = (new_prediction > 0.5)

# Part 4 Evaluating, Improving and tunning ANN

# Evaluating:
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
def build_classifier():
    classifier = Sequential()
    # first hidden layer
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim=11))
    # second hidden layer
    classifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))
    # output layer
    classifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
    # compiling the ANN
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier

classifier = KerasClassifier(build_fn=build_classifier, batch_size=10, epochs=10)
# TODO: Fix n_jobs on Macos
accuracies = cross_val_score(estimator=classifier, X=X_train, y=y_train, cv=10) # n_jobs=-1)
mean = accuracies.mean()
variance = accuracies.std()