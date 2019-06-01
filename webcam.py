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

filename = os.path.join( os.getcwd(), 'webcam_img.jpg')

# initialize the camera
cam = cv2.VideoCapture(0)   # 0 -> index of camera
s, img = cam.read()
if s:    # frame captured without any errors
    # cv2.namedWindow("cam-test")
    # cv2.imshow("cam-test",img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("cam-test")
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    # cv2.waitKey(1)
    cv2.imwrite(filename,img) #save image

img = cv2.imread(filename)
# print(np.size(img,0))
# print(np.size(img,1))
height = np.size(img,0)
width = np.size(img,1)
ratio = int(200*width/height)
resized = cv2.resize(img, (ratio,200))
# print(np.size(resized,0))
# print(np.size(resized,1))
margin = int(ratio/2 - 100)+1
crop_img = resized[:,margin:ratio-margin+1]
# cv2.imshow("cropped", crop_img)
# cv2.waitKey(0)
# print(np.size(crop_img,0))
# print(np.size(crop_img,1))

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



classifier.load_weights('first_try.h5')
crop_img = crop_img.reshape(-1,200, 200, 3)
preds = classifier.predict(crop_img)
preds = classifier.predict_classes(crop_img)


parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/train/*')
class_names = glob(parent_dir) # Reads all the folders in which images are present
class_names = [get_category_from_filename(x) for x in class_names]
class_names = sorted(class_names) # Sorting them
name_id_map = dict(zip(class_names, range(len(class_names))))
id_name_map = {value: key for key, value in name_id_map.items()}
print("Predicted letter: ",id_name_map[preds[0]])
