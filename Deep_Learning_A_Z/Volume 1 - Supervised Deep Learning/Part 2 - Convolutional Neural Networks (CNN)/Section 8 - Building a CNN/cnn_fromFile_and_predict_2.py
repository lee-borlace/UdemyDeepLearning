from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import numpy as np
from keras.preprocessing import image


# Load the previously-saved model.
filename = 'cnn.h5'
classifier = load_model(filename)

import glob
for filename in glob.iglob('dataset/samples_for_predicting/*.jpg'):

    test_image = image.load_img(filename, target_size = (64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    result = classifier.predict(test_image)

    # LB - we know which number each class corresponds to via training_set.class_indices

    last_slash_index = filename.rfind("\\")
    last_filename_portion = filename[last_slash_index+1:]

    print(result[0][0])

    # LB : Why are these all coming out as exactly 0 or 1? Shouldn't they be fractions?
    if result[0][0] == 1:
        prediction = 'dog'
    else:
        prediction = 'cat'
        
        
        
    print(last_filename_portion + ' : ' + prediction)