# Convolutional Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Building the CNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import initializers
from keras.optimizers import adam

from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

# Initialising the CNN
classifier = Sequential()


classifier.add(Conv2D(64, (4, 4), input_shape = (128, 128, 3), activation = 'relu', kernel_initializer=initializers.uniform(seed=42)))
 
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (4, 4)))
classifier.add(Conv2D(64, (4, 4), activation = 'relu', kernel_initializer=initializers.uniform(seed=42)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (4, 4), activation = 'relu', kernel_initializer=initializers.uniform(seed=42)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))
classifier.add(Conv2D(32, (4, 4), activation = 'relu', kernel_initializer=initializers.uniform(seed=42)))
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
#LB - good practice to choose a power of 2.
classifier.add(Dense(units = 4096, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dropout(rate=0.3, seed=42))
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(BatchNormalization())
classifier.add(Dense(units = 1, activation = 'sigmoid'))

# Compiling the CNN
optimizer_adam = adam(lr = 0.0005, decay = 0.0001)
classifier.compile(optimizer = optimizer_adam, loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images


import tensorflow as tf

from keras.preprocessing.image import ImageDataGenerator

# LB - here we are augmenting the training data by applying random scale, shear, zoom and horizontal flip.
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (128, 128),
                                                 batch_size = 64,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (128, 128),
                                            batch_size = 64,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         steps_per_epoch = 8000,
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps = 2000)

classifier.save(filepath = "cnn_improved_attempt_2.h5")

classifier.model.metrics_names


classifier.val
