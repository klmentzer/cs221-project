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
# crop_img = crop_img.reshape(-1,200, 200, 3)
parent_dir =parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val')
letter_folders = list(set(os.listdir(parent_dir)))
letter_folders = [s for s in letter_folders if s[0]!= '.' and s not in ['test','val','train']]
letter_folders.sort()
acc_dict = {}
results = []
j = 0

# for k in range(3):
#     j=0
#     for letter in letter_folders:
#         print(letter)
#         num = np.random.choice(range(301,601))
#         filename = os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val',letter,str(letter)+str(num)+'.jpg')
#         img = cv2.imread(filename)
#         img = img.reshape(-1,200, 200, 3)
#         preds = classifier.predict(img)
#         print(preds)
#
#         parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val',letter,'*')
#         letter_imgs = glob(parent_dir)
#         for i in range(len(letter_imgs)):
#             img = cv2.imread(letter_imgs[i])
#             to_test[i,:,:,:] = img
#
#         expected = np.tile(preds,300)
#         expected = expected.reshape((300,29))
#         # expected = np.zeros((300,29))
#         # expected[:,5] = np.ones(300)
#         preds = classifier.evaluate(to_test,expected)
#         print(preds)
#         print(classifier.metrics_names)
#         #acc_dict[letter] = preds[1]
#         if k == 0:
#             results.append([letter,preds[1]])
#         else:
#             results[j].append(preds[1])
#         j +=1
#
# print(acc_dict)
# for result in results:
#     print(result)

results = [['A', 0.8633333341280619, 0.8633333341280619, 0.8633333341280619],
['B', 0.39, 0.39, 0.27],
['C', 0.97, 0.97, 0.97],
['D', 0.5533333325386047, 0.5533333325386047, 0.44333333293596905],
['E', 0.42333333353201547, 0.576666665871938, 0.42333333353201547],
['F', 0.9200000007947285, 0.03333333333333333, 0.9200000007947285],
['G', 0.5633333335320155, 0.5633333335320155, 0.4366666658719381],
['H', 0.9666666658719381, 0.9666666658719381, 0.9666666658719381],
['I', 0.74, 0.74, 0.74],
['J', 0.9933333333333333, 0.9933333333333333, 0.9933333333333333],
['K', 0.5533333333333333, 0.43, 0.43],
['L', 0.8933333333333333, 0.8933333333333333, 0.10666666666666667],
['M', 0.573333334128062, 0.573333334128062, 0.31333333333333335],
['N', 0.82, 0.82, 0.82],
['O', 0.7766666674613952, 0.22333333373069764, 0.22333333373069764],
['P', 0.5400000007947287, 0.4600000003973643, 0.4600000003973643],
['Q', 0.38666666666666666, 0.6133333333333333, 0.38666666666666666],
['R', 0.6566666674613952, 0.6566666674613952, 0.6566666674613952],
['S', 0.9866666666666667, 0.9866666666666667, 0.9866666666666667],
['T', 0.88, 0.04666666666666667, 0.88],
['U', 0.6766666674613953, 0.2, 0.06],
['V', 0.7899999992052714, 0.20666666626930236, 0.7899999992052714],
['W', 0.6899999992052714, 0.6899999992052714, 0.6899999992052714],
['X', 0.8966666666666666, 0.8966666666666666, 0.8966666666666666],
['Y', 0.8933333333333333, 0.8933333333333333, 0.8933333333333333],
['Z', 0.8966666666666666, 0.8966666666666666, 0.8966666666666666],
['del', 0.7766666658719381, 0.7766666658719381, 0.7766666658719381],
['nothing', 0.07, 0.6666666666666666, 0.6666666666666666],
['space', 0.556666665871938, 0.16333333333333333, 0.556666665871938]]




labels = [x[0] for x in results]
acc = [max(x[1:]) for x in results]


y_pos = np.arange(len(labels))

fig = plt.figure()
plt.bar(y_pos, acc, align='center', alpha=0.5)
plt.xticks(y_pos, labels,rotation='vertical')
plt.xlabel('Letter')
plt.ylabel('Accuracy')
plt.title("Classification Accuracy by Letter")
fig.tight_layout()


plt.show()

# filename = os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val/A/A301.jpg')
# img = cv2.imread(filename)
# img = img.reshape(-1,200, 200, 3)
# preds = classifier.predict(img)
# print(preds)

# parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val/*')
# class_names = glob(parent_dir)

# print(class_names)

# parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/val/*')
# class_names = glob(parent_dir) # Reads all the folders in which images are present
# class_names = [get_category_from_filename(x) for x in class_names]
# class_names = sorted(class_names) # Sorting them
# name_id_map = dict(zip(class_names, range(len(class_names))))
# id_name_map = {value: key for key, value in name_id_map.items()}
# print("Predicted letter: ",id_name_map[preds[0]])
