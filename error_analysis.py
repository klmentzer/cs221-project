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
from sklearn.metrics import classification_report
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



### TRY ALL OF VAL DATA SET ###

# lett_dir = os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val')
# print(lett_dir)
# test_datagen = image.ImageDataGenerator(rescale=1./255)
#
# test_generator = test_datagen.flow_from_directory(
#         lett_dir,
#         target_size=(200,200),
#         batch_size=32,
#         class_mode='categorical')
# results = classifier.evaluate_generator(generator=test_generator, steps=300,verbose=1)
# print(classifier.metrics_names)
# print(results)

### TRY RANDOM LETTERS ###

parent_dir =parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val')
letter_folders = list(set(os.listdir(parent_dir)))
letter_folders = [s for s in letter_folders if s[0]!= '.' and s not in ['test','val','train']]
acc_dict = {}
results = []
j = 0

temp = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, \
    'I': 8, 'J': 9, 'K': 10, 'L': 11, 'M': 12, 'N': 13, 'O': 14, 'P': 15, \
    'Q': 16, 'R': 17, 'S': 18, 'T': 19, 'U': 20, 'V': 21, 'W': 22, 'X': 23, \
    'Y': 24, 'Z': 25, 'del': 26, 'nothing': 27, 'space': 28}
label_dict = {value: key for key, value in temp.items()}

for letter in letter_folders:
    print(letter)
    num = np.random.choice(range(301,601))
    filename = os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val',letter,str(letter)+str(num)+'.jpg')
    img = cv2.imread(filename)
    plt.figure()
    plt.imshow(img)
    plt.show()
    img = img.reshape(-1,200, 200, 3)
    preds = classifier.predict(img)
    print(preds)
    preds2 = classifier.predict_classes(img)
    #print(preds2)
    print(label_dict[preds2[0]])
