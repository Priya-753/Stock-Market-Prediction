#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 28 13:02:26 2018

@author: Priya
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 27 19:41:28 2018

@author: Priya
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from pandas_datareader import data,wb
import datetime

start = datetime.datetime(2010,1,1)
end = datetime.date.today()
dataset_train = data.DataReader('YHOO','morningstar',start,end)

cols = list(dataset_train)[0:5]
dataset_train = dataset_train[cols]

dataset_train

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

training_set_scaled


sc_predict = MinMaxScaler(feature_range=(0,1))
sc_predict.fit_transform(training_set[:,0:1])


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

reframed = series_to_supervised(training_set_scaled, 60, 1)

print(reframed.head())

values = reframed.values

n_days = 60
n_features = 5
n_train = 2068
train = values[:n_train, :]
test = values[n_train:, :]
n_obs = n_days * n_features
train_X, train_y = train[:, :n_obs], train[:, -n_features]
test_X, test_y = test[:, :n_obs], test[:, -n_features]
print(train_X.shape, len(train_X), train_y.shape)

train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
test_X = test_X.reshape((test_X.shape[0], n_days, n_features))
print(train_X.shape, train_y.shape, test_X.shape, test_y.shape)

from keras.layers import Dropout

#model
model = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(50,input_shape=(train_X.shape[1], train_X.shape[2])))

# Adding a fourth LSTM layer and some Dropout regularisation
#output layer
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')

# fit network
history = model.fit(train_X, train_y, epochs=200, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)

dataset_test = training_set[2128:2148,0:1]
dataset_test.shape

predictions = model.predict(test_X)

#predictions[0] is supposed to predict y_train[19] and so on.
predictions_plot = sc_predict.inverse_transform(predictions[0:])

actual_plot = dataset_test

hfm, = plt.plot(predictions_plot, 'r', label='predicted_stock_price')
hfm2, = plt.plot(actual_plot,'b', label = 'actual_stock_price')
plt.title('Yahoo Stock Price Prediction')
plt.xlabel('Day')
plt.ylabel('Yahoo Close Price')
plt.legend()
plt.show()


model_json = model.to_json()

with open("YHOO.json","w") as json_file:
    json_file.write(model_json)

model.save_weights("YHOO.h5")

print("Saved model to disk")

