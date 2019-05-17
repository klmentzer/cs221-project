'''
First stab at loading data into Keras
'''

from keras.preprocessing import image
from keras import layers
import os

parent_dir =os.path.join( os.getcwd(), '../asl-alphabet/asl_alphabet_train/')
test_dir = parent_dir+"test"
val_dir = parent_dir+"val"
train_dir = parent_dir+"train"

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
