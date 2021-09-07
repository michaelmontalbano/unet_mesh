# Author: Michael Montalbano
# Create U-Network or Autoencoder

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import GaussianNoise, AveragePooling2D, Dropout, BatchNormalization, Convolution2D, Dense, MaxPooling2D, Flatten, BatchNormalization, Dropout, Concatenate, Input, UpSampling2D, Add
from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from sklearn.metrics import mean_squared_error
from metrics_binarized import *
import sys
import numpy as np

# channels should increase as you step down the U-Net
# ratio of rows*columns/channels should go down
def create_uNet(input_shape, nclasses=2, filters=[40,80,160], 
                   lambda_regularization=None, activation='relu',dropout=None,
                   type='binary', optimizer='adam',threshold=0.1):
                
    global thres 
    thres = threshold

    if lambda_regularization is not None:
        lambda_regularization = keras.regularizers.l2(lambda_regularization)
    
    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
    activation = lrelu
    
    tensor_list = []
    input_tensor = Input(shape=input_shape, name="input")

    # First Layer is Batch Normalization
    tensor = BatchNormalization(axis=[1,2])(input_tensor)
    tensor = GaussianNoise(0.1)(tensor)

    tensor = Convolution2D(filters[0],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)

    if dropout is not None:
        tensor = Dropout(dropout)(tensor)

    # tensor = BatchNormalization()(tensor)

    # tensor = GaussianNoise(10)(tensor)
    
    #############################
    tensor_list.append(tensor)
   
    tensor = AveragePooling2D(pool_size=(2,2),
                          strides=(2,2),
                          padding='same')(tensor)
    

    # 30x30
    tensor = BatchNormalization(axis=[1,2])(tensor)
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)

    tensor_list.append(tensor)

    if dropout is not None:
        tensor = Dropout(dropout)(tensor)

    tensor = AveragePooling2D(pool_size=(2,2),
                          strides=(2,2),
                          padding='same')(tensor)

    # 15x15
    tensor = BatchNormalization(axis=[1,2])(tensor)    
    tensor = Convolution2D(filters[2],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)

    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)   

    if dropout is not None:
        tensor = Dropout(dropout)(tensor)  

    # upsample

    # 30x30
    tensor = UpSampling2D(size=2) (tensor) # take 1 pixel and expand it out to 2 x 2
    tensor = BatchNormalization(axis=[1,2])(tensor)
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)

    tensor = Add()([tensor, tensor_list.pop()])    

    # 60x60
    tensor = UpSampling2D(size=2) (tensor) # take 1 pixel and expand it out to 2 x 2
    tensor = BatchNormalization(axis=[1,2])(tensor)
    tensor = Convolution2D(filters[0],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)

    tensor = Add()([tensor, tensor_list.pop()])

    tensor = BatchNormalization(axis=[1,2])(tensor)

    #############################   
    if type == 'binary':
        output_tensor = Convolution2D(1,
                          kernel_size=(1,1),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation='sigmoid',name='output')(tensor)
    if type == 'regression' or type == 'custom':
        output_tensor = Convolution2D(nclasses,
                          kernel_size=(1,1),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation,name='output')(tensor)

    if optimizer == 'adam':
        opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                 epsilon=None, decay=0.0, amsgrad=False)
    if optimizer == 'SGD':
        opt = keras.optimizers.SGD(lr = 1e-4, momentum=0.1, 
                                nesterov=False, name='SGD')
    if optimizer == 'RMSprop':
        opt = keras.optimizers.RMSprop(lr=0.00001)

    model = Model(inputs=input_tensor, outputs=output_tensor)

    if type == 'custom':
        model.compile(loss=customMSE,optimizer=opt,
                  metrics=['mse'])   

    if type == 'regression':
        model.compile(loss='mse',optimizer=opt,
                  metrics=['mse'])
    if type == 'binary':
        model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt,
                      metrics=[tf.keras.metrics.MeanSquaredError()])
    if type == 'categories':
        model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,
                        metrics=tf.keras.metrics.CategoricalCrossentropy())
    
    #model.compile(loss='mse',optimizer="adam",
    #             metrics=[MyBinaryAccuracy(),
    #                       MyAUC()])
    
    return model

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

def customMSE(y_true, y_pred):
    '''
    Correct predictions of 0 do not affect performance.
    Performance is affected only by predictions where y_true > 20
    '''
    loss = tf.square(y_true - y_pred)
    # ignore elements where BOTH y_true & y_pred < 0.1
    mask = tf.cast(tf.logical_or(y_true >= thres, y_pred >= thres) ,tf.float32)
    loss *= mask
    return loss

def alt_unet(input_shape,nclasses=1,dropout=None,l=None):

    input_tensor = Input(shape=input_shape, name="input")


    x = layers.Conv2D(64, 1, strides=2, padding="same")(input_tensor)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x

    for filters in [128]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.AveragePooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same", kernel_regularizer=l2(l), bias_regularizer=l2(l))(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [128, 64]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_regularizer=l2(l), bias_regularizer=l2(l))(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same", kernel_regularizer=l2(l), bias_regularizer=l2(l))(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same",kernel_regularizer=l2(l), bias_regularizer=l2(l))(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    output_tensor = layers.Conv2D(nclasses, 1, activation="relu", padding="same",name='output')(x)

    opt = keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999,
                                 epsilon=None, decay=0.0, amsgrad=False)

    # Define the model
    model = keras.Model(inputs=input_tensor, outputs=output_tensor)

    model.compile(optimizer=opt, loss='mse')

    return model


def keras_custom_loss_function(y_true, y_pred):
    # custom loss function - penalize early on, otherwise leave alone
    loss = np.mean(np.sum(np.square((y_true - y_pred)/10)))
    return loss


def create_seq(input_shape, nclasses, filters=[30,45,60], 
                   lambda_regularization=None, activation='elu'):
    # A sequential model for semantic reasoning

    if lambda_regularization is not None:
        lambda_regularization = keras.regularizers.l2(lambda_regularization)
    
    tensor_list = []
    input_tensor = Input(shape=input_shape, name="input")

    # 256x256
        
    tensor = Convolution2D(filters[0],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(input_tensor)
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    


    tensor = Convolution2D(filters[2],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Convolution2D(filters[2],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)
    
    tensor = Convolution2D(filters[1],
                          kernel_size=(3,3),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation=activation)(tensor)

    #############################        
    output_tensor = Convolution2D(nclasses,
                          kernel_size=(1,1),
                          padding='same', 
                          use_bias=True,
                          kernel_initializer='random_uniform',
                          bias_initializer='zeros',
                          kernel_regularizer=lambda_regularization,
                          activation='sigmoid',name='output')(tensor)
    
    

    model = Model(inputs=input_tensor, outputs=output_tensor)
    opt = keras.optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, 
                                    epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=opt,
                 metrics=["binary_accuracy"])
    
    return model
