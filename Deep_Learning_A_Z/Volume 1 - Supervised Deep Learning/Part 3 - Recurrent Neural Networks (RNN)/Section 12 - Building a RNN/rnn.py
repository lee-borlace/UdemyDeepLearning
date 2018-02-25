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
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)


# Part 3 - Making the predictions and visualising the results

