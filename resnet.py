# import the necessary packages
from keras.layers.normalization import BatchNormalization
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D
from keras.layers import BatchNormalization,Flatten, Conv2D, AveragePooling2D, MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Dense
from keras.layers import Flatten, Add
from keras.layers import Input
from keras.models import Model
from keras.layers import add
from keras.regularizers import l2
from keras import backend as K

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

    # def residual_module(self, data, K, stride, chanDim, red=False, reg=0.0001, bnEps=2e-5, bnMom=0.9):
    #     # the shortcut branch of the ResNet module should be
    #     # initialize as the input (identity) data
    #     shortcut = data
    
    #     # the first block of the ResNet module are the 1x1 CONVs
    #     bn1 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(data)
    #     act1 = Activation("relu")(bn1)
    #     conv1 = Conv2D(int(K * 0.25), (1, 1), use_bias=False, 
    #         kernel_regularizer=l2(reg))(act1)    
        
    #     # the second block of the ResNet module are the 3x3 CONVs
    #     bn2 = BatchNormalization(axis=chanDim, epsilon=bnEps, momentum=bnMom)(conv1)
    #     act2 = Activation("relu")(bn2)
    #     conv2 = Conv2D(int(K * 0.25), (3, 3), strides=stride, padding="same", use_bias=False, 
    #         kernel_regularizer=l2(reg))(act2)

    #     # the third block of the ResNet module is another set of 1x1 CONVs
    #     bn3 = BatchNormalization(axis=chanDim, epsilon=bnEps,momentum=bnMom)(conv2)
    #     act3 = Activation("relu")(bn3)
    #     conv3 = Conv2D(K, (1, 1), use_bias=False, kernel_regularizer=l2(reg))(act3)

    #     # if we are to reduce the spatial size, apply a CONV layer to the shortcut
    #     if red:
    #         shortcut = Conv2D(K, (1, 1), strides=stride, use_bias=False, 
    #             kernel_regularizer=l2(reg))(act1)

    #     # add together the shortcut and the final CONV
    #     x = add([conv3, shortcut])

    #     # return the addition as the output of the ResNet module
    #     return x

    # def build(self, width, height, depth, classes, stages, filters, 
	#     reg=0.0001, bnEps=2e-5, bnMom=0.9):
    #     # initialize the input shape to be "channels last" and the
    #     # channels dimension itself
    #     inputShape = (height, width, depth)
    #     chanDim = -1

    #     # if we are using "channels first", update the input shape
    #     # and channels dimension
    #     if K.image_data_format() == "channels_first":
    #         inputShape = (depth, height, width)
    #         chanDim = 1
    #     # set the input and apply BN
    #     inputs = Input(shape=inputShape)
    #     x = BatchNormalization(axis=chanDim, epsilon=bnEps,
    #         momentum=bnMom)(inputs)

    #     # apply CONV => BN => ACT => POOL to reduce spatial size
    #     x = Conv2D(filters[0], (5, 5), use_bias=False,
    #         padding="same", kernel_regularizer=l2(reg))(x)
    #     x = BatchNormalization(axis=chanDim, epsilon=bnEps,
    #         momentum=bnMom)(x)
    #     x = Activation("relu")(x)
    #     x = ZeroPadding2D((1, 1))(x)
    #     x = MaxPooling2D((3, 3), strides=(2, 2))(x)
    #     # loop over the number of stages

    #     for i in range(0, len(stages)):
    #         # initialize the stride, then apply a residual module
    #         # used to reduce the spatial size of the input volume
    #         stride = (1, 1) if i == 0 else (2, 2)
    #         x = self.residual_module(x, filters[i + 1], stride,
    #                 chanDim, red=True, bnEps=bnEps, bnMom=bnMom)

    #         # loop over the number of layers in the stage
    #         for j in range(0, stages[i] - 1):
    #             # apply a ResNet module
    #             x = self.residual_module(x, filters[i + 1],
    #                 (1, 1), chanDim, bnEps=bnEps, bnMom=bnMom)
    #     # apply BN => ACT => POOL
    #     x = BatchNormalization(axis=chanDim, epsilon=bnEps,
    #             momentum=bnMom)(x)
    #     x = Activation("relu")(x)
    #     x = MaxPooling2D((2, 2))(x)

    #         # softmax classifier
    #     x = Flatten()(x)
    #     x = Dense(classes, kernel_regularizer=l2(reg))(x)
    #     x = Activation("softmax")(x)
        
    #     # create the model
    #     model = Model(inputs, x, name="resnet")
        
    #     # return the constructed network architecture
    #     return model