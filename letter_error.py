import cv2
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
# crop_img = crop_img.reshape(-1,200, 200, 3)
parent_dir =parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val')
letter_folders = list(set(os.listdir(parent_dir)))
letter_folders = [s for s in letter_folders if s[0]!= '.' and s not in ['test','val','train']]
acc_dict = {}
results = []
j = 0
for letter in letter_folders:
    print(letter)
    num = np.random.choice(range(301,601))
    filename = os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val',letter,str(letter)+str(num)+'.jpg')
    img = cv2.imread(filename)
    img = img.reshape(-1,200, 200, 3)
    preds = classifier.predict(img)
    print(preds)

    parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val',letter,'*')
    letter_imgs = glob(parent_dir)
    for i in range(len(letter_imgs)):
        img = cv2.imread(letter_imgs[i])
        to_test[i,:,:,:] = img

    expected = np.tile(preds,300)
    expected = expected.reshape((300,29))
    # expected = np.zeros((300,29))
    # expected[:,5] = np.ones(300)
    preds = classifier.evaluate(to_test,expected)
    print(preds)
    print(classifier.metrics_names)
    acc_dict[letter] = preds[1]
    results.append([letter,preds[1]])
    j +=1

print(acc_dict)
for result in results:
    print(result)

# filename = os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val/A/A301.jpg')
# img = cv2.imread(filename)
# img = img.reshape(-1,200, 200, 3)
# preds = classifier.predict(img)
# print(preds)

parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val/*')
class_names = glob(parent_dir)

# print(class_names)

# parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val/*')
# class_names = glob(parent_dir) # Reads all the folders in which images are present
# class_names = [get_category_from_filename(x) for x in class_names]
# class_names = sorted(class_names) # Sorting them
# name_id_map = dict(zip(class_names, range(len(class_names))))
# id_name_map = {value: key for key, value in name_id_map.items()}
# print("Predicted letter: ",id_name_map[preds[0]])
