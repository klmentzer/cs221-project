import os
import matplotlib.pyplot as plt
import numpy as np
from keras.preprocessing import image
from keras.models import Sequential
# from keras import layers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras import optimizers
from segment_data import get_category_from_filename
from glob import glob
os.environ['KMP_DUPLICATE_LIB_OK']='True'

classifier = Sequential()

classifier.add(Conv2D(32, (5, 5), input_shape = (200, 200, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(32, (3, 3), input_shape = (200, 200, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Conv2D(64, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

classifier.add(Flatten())
classifier.add(Dense(64))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))
classifier.add(Dense(29))
classifier.add(Activation('softmax'))


to_test= np.zeros((300,200, 200, 3))
classifier.load_weights('first_try.h5')
classifier.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

test_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/test')
test_datagen = image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=(200,200),
        batch_size=32,
        class_mode='categorical')

results = classifier.evaluate_generator(generator=test_generator, steps=300,verbose=1)
print(classifier.metrics_names)
print(results)
