# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('/Users/Priya/Desktop/Deep_Learning_A_Z/SDL/RNN/B-RNN/Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
#standard shape for input to rnn
# 3d array batch size,time step, indicators
#indicators- closing price/stock of other companies i.e correlation factor

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential #sequnce of layers
from keras.layers import Dense#output layer
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()
#an on=bject of class sequential because it will create a sequence of layers

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
#Lstm layer added to the regressor obj
#units-number of neurons in the lstm layer (3-5 means low dim) we need high dim to capture complex features
#return_sequences = True cause stack of lstm for last lstm layer it is set to flase we dont have to explicitly do so cause it is the default value
#input_shape - last to dims of the X_train
regressor.add(Dropout(0.2))
#20% of neurons will get dropped out during every iteration of training duting fwd prop and bckwd prop

#adding extra lstm layers to improve dimesionality

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))


# Adding a fourth LSTM layer and some Dropout regularisation
#no return_sequnces cause it is the last lstm layer. default return_sequences = false
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))


# Adding the output layer
# dense class because this output layer is fully connnected to all the neurons of the previous lstm layer
#units =1 cause we are predicting only one value i.e the stock price of next day
regressor.add(Dense(units = 1))


# Compiling the RNN
#compile is another method of sequential class
#usually RMSprop optimiser is used but adam can also be used as it updates the weights relevantly
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

#so far highly robust rnn
#we have to connect it to the training set
# Fitting the RNN to the Training set
#connects and executes certain number of epochs chosen in the fit method
#X_train is input
#y_train is the ground truth
#epochs is iterations(25 epochs no convergence)(100 epochs for 5 years data)
#batch_size- trained on batches of observed not for every stock price. now =32 thus weight updated not for every fwd prop and error update on every back prop but once every 32 stock prices

regressor.fit(X_train, y_train, epochs = 100, batch_size = 32)

#after every 10 epochs clearly the loss function reduced. loss in the end was 0.0015. if we further reduce the loss then it results in overfitting so must not try to reduce it further

# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price of 2017
# we need the previous 60 days to predict the next day's price. these 60 days may come from the test set or training set or both
#hence we need to concatenate the two
# We must not concatenate the scaled input and scale the dataset_test . We must not ever change the output values. So we concatenate the actual input dataset_train  and the dataset_test and get the input of each prediction  i.e 60 previous stock prices at each time t and then scale thus scaling onlt the input and not changing the values and leads to good results. We must scale because we trained using scaled values and must scale with the same method i.e normalisation with the sc object
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis = 0)
# 1st args which to concatenate and and axis =0 which tells us along which axis . Here along the vertical axis i.e concatenate the lines. if horizontal axis=1
# input of 60 previous days into numpy array. lower bound is the 60 days before the first day in january. upper bound is the last input needed which will be the last value of data_set total. Lower bound =len(dataset_total) - len(dataset_test) - 60, where len(dataset_total) - len(dataset_test) is the index of first day in test data i.e january, till end thus ':'
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
#as we didnt use the iloc method it wont be shaped the right way. so we must reshape with -1,1 as args
inputs = inputs.reshape(-1,1)
#scaling cause rnn was trained with scaled inputs. We need not use fit_transform because sc object was already fitted to the training set. So no need to fit.
inputs = sc.transform(inputs)

#input array for test set. We dont need ground truth so no y_test
X_test = []
#80 because only 20 days in the test set. It always starts from 60 because the timesteps is 60
for i in range(60, 80):
    X_test.append(inputs[i-60:i, 0]) #0 is the column i.e open price which we are using to predict
X_test = np.array(X_test)

# now must change to format expected by rnn for training. format is the 3d format from above
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

#now can predict
predicted_stock_price = regressor.predict(X_test)
#now must inverse the scaling that we did
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend()
plt.show()

#there is a spike in the actual price which was unable to be predicted by our model because it is a fast non linear change. Brownian Motion Concept of financial eng - Non linear changes cannot be predicted by our model because such changes are not dependent on the previous days data. However the model responds okay to smooth changes
