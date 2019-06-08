'''
Classification with ResNet
'''

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
# from resnet import ResNet50
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
import tensorflow as tf
import sys
from kerasinceptionV4master.inception_v4 import create_model
import os


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    with tf.device('/device:GPU:0'):
        INPUT_SHAPE = (299, 299, 3)
        NUM_CLASSES = 29

        # set directory names
        parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/')
        test_dir = parent_dir + "test"
        val_dir = parent_dir + "val"
        train_dir = parent_dir + "train"

        # resnet = ResNet50(classes=NUM_CLASSES, input_shape=INPUT_SHAPE)
        # model = resnet.build()
        base_model = ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=(200, 200, 3), pooling=None, classes=1000)

        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.7)(x)
        predictions = Dense(NUM_CLASSES, activation= 'softmax')(x)
        model = Model(inputs = base_model.input, outputs = predictions)
        # model = Conv2D(32, (5, 5), input_shape = (299, 299, 3), activation='relu')(model)
        # model = MaxPooling2D(pool_size=(2, 2))(model)
        
        # model = Conv2D(32, (3, 3), input_shape = (299, 299, 3), activation='relu')(model)
        # model = MaxPooling2D(pool_size=(2, 2))(model)
        
        # model = Conv2D(64, (3, 3), activation='relu')(model)
        # model = MaxPooling2D(pool_size=(2, 2))(model)
        
        # model = Flatten()(model)
        # model = Dense(64)(model)
        # model = Activation('relu')(model)
        # model = Dropout(0.5)(model)
        # model = Dense(29)(model)
        # model = Activation('softmax')(model)
        model.compile(loss='categorical_crossentropy',
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
                target_size=(200,200),
                batch_size=32,
                class_mode="categorical")

        validation_generator = test_datagen.flow_from_directory(
                val_dir,
                target_size=(200,200),
                batch_size=32,
                class_mode='categorical')


        model.fit_generator(train_generator,
                steps_per_epoch=2000,
                epochs=40,
                validation_data=validation_generator,
                validation_steps=800)

        model.save_weights('resnet_weights.h5')

if __name__ == '__main__':
    main()