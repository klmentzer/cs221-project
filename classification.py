'''
First stab at loading data into Keras
'''

from keras.preprocessing import image
from keras.models import Sequential
# from keras import layers
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
from keras import optimizers
from resnet import ResNet
import sys
sys.path.insert(0, 'kerasinceptionV4master/inception_v4')
from kerasinceptionV4master.inception_v4 import create_model
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

#with device('/gpu:0'):

parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/')
test_dir = parent_dir+"test"
val_dir = parent_dir+"val"
train_dir = parent_dir+"train"

# classifier = create_model(num_classes=29, dropout_prob=0.2, weights=None, include_top=True)#Sequential()
# for layer in classifier.layers[:9*len(classifier.layers)//10]:
#     layer.trainable = False

resnet = ResNet()
classifier = resnet.build(299, 299, 3, 29, (3, 4, 6), filters=(64, 128, 256, 512))

# print(len(classifier.layers))
#
# classifier.add(Conv2D(32, (5, 5), input_shape = (200, 200, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
# classifier.add(Conv2D(32, (3, 3), input_shape = (200, 200, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
# classifier.add(Conv2D(64, (3, 3), activation='relu'))
# classifier.add(MaxPooling2D(pool_size=(2, 2)))
#
# classifier.add(Flatten())
# classifier.add(Dense(64))
# classifier.add(Activation('relu'))
# classifier.add(Dropout(0.5))
# classifier.add(Dense(29))
# classifier.add(Activation('softmax'))

# rmsprop = optimizers.RMSprop(lr=0.01, rho=0.7, epsilon=1e-8, decay=0.0)
#
classifier.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
# create tool to input image data
train_datagen = image.ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        validation_split=0.1)

test_datagen = image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(299,299),
        batch_size=32,
        class_mode="categorical")

validation_generator = test_datagen.flow_from_directory(
        val_dir,
        target_size=(299,299),
        batch_size=32,
        class_mode='categorical')
# print(classifier.summary())

classifier.fit_generator(train_generator,
        steps_per_epoch=2000,
        epochs=5,
        validation_data=validation_generator,
        validation_steps=800)

classifier.save_weights('first_try.h5')
    # layers.Input(name='the_input', shape=(200,200), dtype='float32')
    # labels = layers.Input(name='the_labels',
    #                    shape=[img_gen.absolute_max_string_len], dtype='float32')
    #
    # model.fit_generator(
    #         train_generator,
    #         steps_per_epoch=2000,
    #         epochs=50,
    #         validation_data=validation_generator,
    #         validation_steps=800)
