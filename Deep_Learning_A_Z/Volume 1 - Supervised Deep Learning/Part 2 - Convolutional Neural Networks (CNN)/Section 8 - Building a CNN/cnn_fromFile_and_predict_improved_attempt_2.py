from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

filename = 'cnn_improved_attempt_2.h5'
classifier = load_model(filename)

predict_datagen = ImageDataGenerator(rescale = 1./255)

predict_set = predict_datagen.flow_from_directory('dataset/predict_set',
                                            target_size = (128, 128),
                                            class_mode = None,
                                            seed=None,
                                            shuffle=False)

predictions = classifier.predict_generator(predict_set, verbose=1)

file_names = predict_set.filenames

for index, filename in enumerate(file_names, start=0):
    
    if predictions[index] >= 0.5:
        prediction = 'dog'
    else:
        prediction = 'cat'
        
    print(filename + " is a picture of a " + prediction + " (" + str(predictions[index]) + ")")
        