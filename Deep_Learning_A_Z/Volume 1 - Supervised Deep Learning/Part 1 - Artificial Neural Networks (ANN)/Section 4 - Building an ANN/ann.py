# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras


## What devices available?
#from tensorflow.python.client import device_lib
#print(device_lib.list_local_devices())

# TODO : Reinstall 


############################################################################################################################
# Part 1 - Data Preprocessing
############################################################################################################################

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# LB - turn the country text category into a range of values.
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

# LB - same for gender.
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# LB - Country is now a number with 3 values. We need to change this into 3 separate dummy variables, each of which is either 0 or 1. I believe
# we don't need to do that for gender because it is already only 0 or 1.
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

# LB - this part needs some classification. Here we are ditching the 1st of the dummy variables to avoid the "dummy variable" trap. I don't
# know what that is or why ditching one of the categories doesn't cause an issue.
# Update - See http://www.algosome.com/articles/dummy-variable-trap-regression.html. 
# I think that if you have more dummy variables that are multicollinear (highly correlated - one value can be predicted from others),
# then this causes issues with trying to invert singular matrices. In our case if we know that the country is not one of the other 2, then
# it must be the 3rd.
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
# LB - we're going to test on 20% of the data, which means training on 80%.
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
# LB - get all the features into a similar scale.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)




#############################################################################################################################
## Part 2 - Now let's make the ANN!
#############################################################################################################################
#
## Importing the Keras libraries and packages
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#
## Initialising the ANN
#classifier = Sequential()
#
## LB - add some layers. 
## relu means rectifier linear unit.  
## uniform initializer is for initial input weights.
#
## Adding the input layer and the first hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#
## Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#
## Adding the output layer. If we had more than 2 categories we would use softmax instead of sigmoid.
#classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#
## Compiling the ANN
#
##LB - adam optimizer is extension of stochastic gradient descent. See https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/.
## We use binary cross entropy because output node is sigmoid. Could have used category_crossentropy if more than 2 output classes.
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
## Fitting the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)




#############################################################################################################################
## Part 3 - Making predictions and evaluating the model
#############################################################################################################################
#
## Predicting the Test set results
#y_pred = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
#
#
## Making the Confusion Matrix
## LB - see http://www.dataschool.io/simple-guide-to-confusion-matrix-terminology/.
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#print(cm)

#############################################################################################################################
## LB - predict one-off sample.
#############################################################################################################################
#
##Geography: France (0 0 for dummy variables)
##Credit Score: 600
##Gender: Male
##Age: 40 years old
##Tenure: 3 years
##Balance: $60000
##Number of Products: 2
##Does this customer have a credit card ? Yes
##Is this customer an Active Member: Yes
##Estimated Salary: $50000
#X_sample = np.array([[0,0,600,1,40,3,60000,2,1,1,50000]])
#X_sample = sc.fit_transform(X_sample)
#predicted = classifier.predict(X_sample)


############################################################################################################################
# LB - K-fold cross validation. The purpose of this is to get a good measure of performance.
############################################################################################################################
#from keras.wrappers.scikit_learn  import KerasClassifier
#from sklearn.model_selection import cross_val_score
#from keras.layers import Dropout
#from keras.models import Sequential
#from keras.layers import Dense
#def build_classifier():
#    classifier = Sequential()
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#    classifier.add(Dropout(p = 0.1))
#    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#    classifier.add(Dropout(p = 0.1))
#    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#    return classifier
#
#classifier2 = KerasClassifier(build_fn=build_classifier, batch_size = 10, epochs = 100)
#
## cv is number of folds, n_jobs indicates run on all CPUs.
#
## LB - TODO : This parallel bit fails with n_jobs = -1. Need to work out how to parralel it.
## accuracies = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv=10, n_jobs = -1)
#accuracies = cross_val_score(estimator = classifier2, X = X_train, y = y_train, cv=10, n_jobs=1)
#
#print (accuracies.mean())
#print (accuracies.std())


############################################################################################################################
# LB - Improving with dropout regularization
############################################################################################################################
#import keras
#from keras.models import Sequential
#from keras.layers import Dense
#from keras.layers import Dropout
#
#
## Initialising the ANN
#classifier = Sequential()
#
## LB - add some layers. 
## relu means rectifier linear unit.  
## uniform initializer is for initial input weights.
#
## Adding the input layer and the first hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
#classifier.add(Dropout(p = 0.1))
#
## Adding the second hidden layer
#classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#classifier.add(Dropout(p = 0.1))
#
## Adding the output layer. If we had more than 2 categories we would use softmax instead of sigmoid.
#classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#
## Compiling the ANN
#
##LB - adam optimizer is extension of stochastic gradient descent. See https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/.
## We use binary cross entropy because output node is sigmoid. Could have used category_crossentropy if more than 2 output classes.
#classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
#
## Fitting the ANN to the Training set
#classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)


############################################################################################################################
# LB - Improving with hyperparameter tuning
############################################################################################################################
from keras.wrappers.scikit_learn  import KerasClassifier
from keras.models import Sequential
from sklearn.model_selection import GridSearchCV
from keras.layers import Dropout
from keras.layers import Dense

# Note we changed def to pass in optimizer and dropout so we can tune that param.
def build_classifier(optimizer, dropout):
    classifier = Sequential()
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))
    classifier.add(Dropout(p = dropout))
    classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
    classifier.add(Dropout(p = dropout))
    classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
    classifier.compile(optimizer = optimizer, loss = 'binary_crossentropy', metrics = ['accuracy'])
    return classifier

classifier3 = KerasClassifier(build_fn=build_classifier)

# Specify the different parameters we want to try in different combos.
parameters = {'batch_size': [25, 32, 64, 128, 256, 512],
              'epochs': [100,200,300,400,500],
              'optimizer': ['adam'],
              'dropout' : [0.1, 0.2]}

grid_search = GridSearchCV(estimator = classifier3,
                           param_grid = parameters,
                           scoring = 'accuracy',
                           cv = 10)

grid_search = grid_search.fit(X_train, y_train)

# Get best results.
best_parameters = grid_search.best_params_ 
best_accuracy = grid_search.best_score_



