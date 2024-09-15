#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 24 10:29:19 2018

@author: Priya
"""


# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard


# Importing the training set
#data frame
dataset_train = pd.read_csv('/Users/Priya/Desktop/Deep_Learning_A_Z/SDL/RNN/B-RNN/Google_Stock_Price_Train.csv')
#numpy array

cols = list(dataset_train)[1:5]

dataset_train = dataset_train[cols]

#dataset_train
dataset_train = dataset_train[cols].astype(str)

for i in cols:
    for j in range(0,len(dataset_train)):
        dataset_train[i][j] = dataset_train[i][j].replace(",","")

dataset_train = dataset_train.astype(float)

training_set = dataset_train.as_matrix()

from sklearn.preprocessing import MinMaxScaler

sc = MinMaxScaler(feature_range=(0, 1))
training_set_scaled = sc.fit_transform(training_set)
 
sc_predict = MinMaxScaler(feature_range=(0,1))
 
sc_predict.fit_transform(training_set[:,0:1])

X_train = []
y_train = []
 
n_future = 20  # Number of days you want to predict into the future
n_past = 60  # Number of past days you want to use to predict the future

for i in range(n_past, len(training_set_scaled) - n_future + 1):
    X_train.append(training_set_scaled[i - n_past:i, 0:5])
    y_train.append(training_set_scaled[i+n_future-1:i + n_future, 0])
    
X_train, y_train = np.array(X_train), np.array(y_train)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

regressor = Sequential()

# Adding fist LSTM layer and Drop out Regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(n_past, 4)))
regressor.add(Dropout(.2))

 # Part 3 - Adding more layers
 
# Adding 2nd layer with some drop out regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(.2))
 
# Adding 3rd layer with some drop out regularization
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(.2))
 
# Adding 4th layer with some drop out regularization
regressor.add(LSTM(units=50, return_sequences=False))
regressor.add(Dropout(.2))
 
# Output layer
regressor.add(Dense(units=1, activation='sigmoid'))
 
# Compiling the RNN
regressor.compile(optimizer='adam', loss="binary_crossentropy")  # Can change loss to mean-squared-error if you require.
 
# Fitting RNN to training set using Keras Callbacks. Read Keras callbacks docs for more info.
 
es = EarlyStopping(monitor='val_loss', min_delta=1e-10, patience=20, verbose=1)
rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
mcp = ModelCheckpoint(filepath='weights.h5', monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True)
tb = TensorBoard('logs')
 
history = regressor.fit(X_train, y_train, shuffle=True, epochs=100,
                        callbacks=[es, rlr,mcp, tb], validation_split=0.2, verbose=1, batch_size=64)
 

#actual stock price
#Predicting on the training set  and plotting the values. the test csv only has 20 values and
# "ideally" cannot be used since we use 60 timesteps here.
 
predictions = regressor.predict(X_train)
predictions
#predictions[0] is supposed to predict y_train[19] and so on.
predictions_plot = sc_predict.inverse_transform(predictions[0:-20])
actual_plot = sc_predict.inverse_transform(y_train[19:-1])
predictions_plot
actual_plot
 
hfm, = plt.plot(predictions_plot, 'r', label='predicted_stock_price')
hfm2, = plt.plot(actual_plot,'b', label = 'actual_stock_price')
 
plt.legend(handles=[hfm,hfm2])
plt.title('Predictions vs Actual Price')
plt.xlabel('Sample index')
plt.ylabel('Stock Price')
plt.savefig('graph.png', bbox_inches='tight')
plt.show()
plt.close()