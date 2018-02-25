# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Part 2 - Building the RNN
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')

# LB - all rows, but only column 1. Need to specify range 1:2 because we need an array of arrays.
training_set = dataset_train.iloc[:, 1:2].values 

# Feature Scaling
# LB - we are going to use normalization. Xnorm = (x - min(x)) / max(x) - min(x)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1)) # LB - values will be between 0 and 1.
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output. 
# LB - the number of timesteps to use is very important. 60 comes from prior experimentation. 60 is based on 20 business days per
# month. So each day's prediction is based on 3 months prior data.
# Each X is 60 days of prior prices.
# Each y is the price for that day.
X_train = []
y_train = []

# LB - start at the 60th item because we can't go back 60 days from any earlier point.
for i in range(60, 1258):
    X_train.append(training_set_scaled[i-60:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
# LB - adding a dimension. We start with a 1198 x 60 array (i.e. X_train.shape[0] by X_train.shape[1]) .
# Our new matrix shape will be 1198 x 60 x 1. If we had other info e.g. another share price which correlates, this could be
# added here as the last dimension.
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

# Part 3 - Making the predictions and visualising the results

