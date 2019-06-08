# import the necessary packages
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization,Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from tensorflow.python.keras.layers.core import Activation
from tensorflow.keras.layers.core import Dense
from tensorflow.keras.layers import Flatten, Add
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import add
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
import tensorflow as tf

class ResNet50():
    def __init__(self, classes, input_shape):
        self.classes = classes
        self.input_shape = input_shape
        self.strides = (1,1)

    
    def identity_block(self, data, filters, stage, block):
        '''
        Creates an identity block in the ResNet. Each block has three Conv2D layers.
        The shortcut path is simply the identity.

        :param data: The input data
        :param filters: A tuple of the number of filters to be used at each layer
        :param stage: Integer indicating the stage of the ResNet we're in
        :param block: Character indicating which block of the stage we're in

        :return: x, output of identity block
        '''

        # identity: keep original data
        shortcut = data

        f1, f2, f3 = filters

        # first layer of the block, uses f1 and kernel size of (1,1)
        data = Conv2D(filters=f1, strides=self.strides, kernel_size=(1,1),
                      input_shape=self.input_shape, padding='valid')(data)
        data = BatchNormalization(axis=3)(data)
        data = Activation('relu')(data)

        # second layer, uses f2 and kernel size of (3, 3); padding = 'same' to keep output
        # dimensions the same as input
        data = Conv2D(filters=f2, strides=self.strides, kernel_size=(3,3),
                      input_shape=self.input_shape, padding='same')(data)
        data = BatchNormalization(axis=3)(data)
        data = Activation('relu')(data)

        # third layer, uses f3 and kernel size of (1,1)
        data = Conv2D(filters=f3, strides=self.strides, kernel_size=(1,1),
                      input_shape=self.input_shape, padding='valid')(data)
        data = BatchNormalization(axis=3)(data)

        # add shortcut to data then apply relu activation function
        data = Add()([data, shortcut])
        data = Activation('relu')(data)

        return data

    
    def conv_block(self, data, filters, stage, block):
        '''
        Creates a convolutional block in the ResNet. Each block has three Conv2D layers.
        The shortcut path is one Conv2D, batch normalized layer.

        :param data: The input data
        :param filters: A tuple of the number of filters to be used at each layer
        :param stage: Integer indicating the stage of the ResNet we're in
        :param block: Character indicating which block of the stage we're in
        '''
        f1, f2, f3 = filters

        # use f3 for shortcut to make sure its output matches the size of desired final output
        shortcut = data
        shortcut = Conv2D(filters=f3, strides=self.strides, kernel_size=(1,1),
                      input_shape=self.input_shape, padding='valid')(shortcut)
        shortcut = BatchNormalization(axis=3)(shortcut)

        # first layer of the block, uses f1 and kernel size of (1,1)
        data = Conv2D(filters=f1, strides=self.strides, kernel_size=(1,1),
                      input_shape=self.input_shape, padding='valid')(data)
        data = BatchNormalization(axis=3)(data)
        data = Activation('relu')(data)

        # second layer, uses f2 and kernel size of (3, 3); padding = 'same' to keep output
        # dimensions the same as input
        data = Conv2D(filters=f2, strides=self.strides, kernel_size=(3,3),
                      input_shape=self.input_shape, padding='same')(data)
        data = BatchNormalization(axis=3)(data)
        data = Activation('relu')(data)

        # third layer, uses f3 and kernel size of (1,1)
        data = Conv2D(filters=f3, strides=self.strides, kernel_size=(1,1),
                      input_shape=self.input_shape, padding='valid')(data)
        data = BatchNormalization(axis=3)(data)

        # add shortcut to data then apply relu activation function
        data = Add()([data, shortcut])
        data = Activation('relu')(data)

        return data
    
    def build(self):
        x_input = Input(self.input_shape)
        x = ZeroPadding2D((3,3))(x_input)

        # stage 1: create first layer, batch normalize, and apply activation
        x = Conv2D(64, (7,7), strides=(2,2), input_shape=self.input_shape)(x)
        x = BatchNormalization(axis=3)(x)
        x = Activation('relu')(x)
        x = MaxPooling2D((3,3), strides=(2,2))(x)

        # stage 2: 1 conv block, 2 identity blocks
        filters = (64, 64, 256)
        x = self.conv_block(x, filters, stage=2, block='a')
        x = self.identity_block(x, filters, stage=2, block='b')
        x = self.identity_block(x, filters, stage=2, block='c')

        # stage 3: 1 conv block, 3 identity blocks
        filters = (128, 128, 512)
        x = self.conv_block(x, filters, stage=3, block='a')
        x = self.identity_block(x, filters, stage=3, block='b')
        x = self.identity_block(x, filters, stage=3, block='c')
        x = self.identity_block(x, filters, stage=3, block='d')

        # stage 4: 1 conv block, 5 identity blocks
        filters = (256, 256, 1024)
        x = self.conv_block(x, filters, stage=4, block='a')
        x = self.identity_block(x, filters, stage=4, block='b')
        x = self.identity_block(x, filters, stage=4, block='c')
        x = self.identity_block(x, filters, stage=4, block='d')
        x = self.identity_block(x, filters, stage=4, block='e')

        # stage 5: 1 conv block, 2 identity blocks
        filters = (512, 512, 2048)
        x = self.conv_block(x, filters, stage=5, block='a')
        x = self.identity_block(x, filters, stage=5, block='b')
        x = self.identity_block(x, filters, stage=5, block='c')
        
        # apply average pooling
        x = AveragePooling2D((2,2))(x)

        x = Flatten()(x)
        x = Dense(self.classes, activation='softmax')(x)

        model = Model(inputs=x_input, outputs=x)
        return model