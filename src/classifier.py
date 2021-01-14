#import necessary packages
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Input
from tensorflow.keras. preprocessing import image
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import numpy as np
import cv2
import os

#Data preprocessing specifications/definitions
img_height = 200
img_width = 200
img_batch_size = 7
no_channels = 3

#Define the no. of classes/labels to be generated for each branch
coarse2_classes = 2 #good state, damaged
fine_classes = 4 #rusty, body, door, none

#Normalize/scale the data
train = ImageDataGenerator(rescale = 1/255)

#generate the train and val set labels by loading data from df/csv file
train_c2_set=train.flow_from_dataframe(
dataframe=traindf,
directory="coarse21/Images/",
x_col="Image",
y_col= columns,
subset="training",
batch_size= img_batch_size,
#seed=42,
shuffle=True,
class_mode="multi_output",
target_size=(img_height,img_width))

#define the structure of the Branch-CNN hierarchical multi-output classifier custom learning model

#blocks 1,2 and 3 architecture for the coarse2 branch
#Block 1
input_shape =(img_height,img_width,no_channels)
img_dimension = Input(shape= input_shape, name='input_dim')
#1st convolution and its input tensor
layer = tf.keras.layers.Conv2D(64,(3,3), padding= 'same', activation= 'relu', name='block1_conv1')(img_dimension)
layer = tf.keras.layers.BatchNormalization()(layer)
#2nd convolution
layer = tf.keras.layers.Conv2D(64,(3,3), padding= 'same', activation= 'relu', name='block1_conv2')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#1st pooling
layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block1_pool')(layer)

#Block 2
#3rd convolution
layer = tf.keras.layers.Conv2D(128,(3,3), padding= 'same', activation= 'relu', name='block2_conv1')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#4th convolution
layer = tf.keras.layers.Conv2D(128,(3,3), padding= 'same', activation= 'relu', name='block2_conv2')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#2nd pooling
layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block2_pool')(layer)

#Block 3
#5th convolution
layer = tf.keras.layers.Conv2D(256,(3,3), padding= 'same', activation= 'relu', name='block3_conv1')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#6th convolution
layer = tf.keras.layers.Conv2D(256,(3,3), padding= 'same', activation= 'relu', name='block3_conv2')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#7th convolution
layer = tf.keras.layers.Conv2D(256,(3,3), padding= 'same', activation= 'relu', name='block3_conv3')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#3rd pooling
layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block3_pool')(layer)

#Coarse 2 branch fc layer
c2_branch = tf.keras.layers.Flatten()(layer)
#c2 branch's 1st dense layer
c2_branch = tf.keras.layers.Dense(1024,activation= 'relu', name='c2_dense1')(c2_branch)
c2_branch = tf.keras.layers.BatchNormalization()(c2_branch)
c2_branch = tf.keras.layers.Dropout(0.5)(c2_branch)
#c2 branch's 2nd dense layer
c2_branch = tf.keras.layers.Dense(1024,activation= 'relu', name='c2_dense2')(c2_branch)
c2_branch = tf.keras.layers.BatchNormalization()(c2_branch)
c2_branch = tf.keras.layers.Dropout(0.5)(c2_branch)
#c2 branch's output layer
c2_branch_out = tf.keras.layers.Dense(coarse2_classes, activation= 'softmax', name='c2_last_dense_layer')(c2_branch)

#Block 4
#8th convolution
layer = tf.keras.layers.Conv2D(512,(3,3), padding= 'same', activation= 'relu', name='block4_conv1')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#9th convolution
layer = tf.keras.layers.Conv2D(512,(3,3), padding= 'same', activation= 'relu', name='block4_conv2')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#10th convolution
layer = tf.keras.layers.Conv2D(512,(3,3), padding= 'same', activation= 'relu', name='block4_conv3')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#4rd pooling
layer = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2), name='block4_pool')(layer)

#Block 5
#11th convolution

layer = tf.keras.layers.Conv2D(512,(3,3), padding= 'same', activation= 'relu', name='block5_conv1')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#12th convolution
layer = tf.keras.layers.Conv2D(512,(3,3), padding= 'same', activation= 'relu', name='block5_conv2')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)
#13th convolution
layer = tf.keras.layers.Conv2D(512,(3,3), padding= 'same', activation= 'relu', name='block5_conv3')(layer)
layer = tf.keras.layers.BatchNormalization()(layer)

#Fine branch fc layer
fine_branch = tf.keras.layers.Flatten()(layer)
#fine branch's 1st dense layer
fine_branch = tf.keras.layers.Dense(4096,activation= 'relu', name='fine_dense1')(fine_branch)
fine_branch = tf.keras.layers.BatchNormalization()(fine_branch)
fine_branch = tf.keras.layers.Dropout(0.5)(fine_branch)
#fine branch's 2nd dense layer
fine_branch = tf.keras.layers.Dense(4096,activation= 'relu', name='fine_dense2')(fine_branch)
fine_branch = tf.keras.layers.BatchNormalization()(fine_branch)
fine_branch = tf.keras.layers.Dropout(0.5)(fine_branch)
#fine branch's output layer
fine_branch_out = tf.keras.layers.Dense(fine_classes, activation= 'softmax', name='fine_last_dense_layer')(fine_branch)

#Puting the learning model together
model = tf.keras.Model(inputs= img_dimension, outputs= [c2_branch_out, fine_branch_out], name= 'edp_eb_hierarchy')

#define the learning model's loss function, the optimizer, its learning rate and observable metrics.
#then compile 
model.compile(loss= 'sparse_categorical_crossentropy',
             optimizer = Adam(lr=1e-03),
             metrics =['accuracy'])

#fit the model to the data, and define no. of epochs, etc, and start training!!!
model_fit = model.fit(train_c2_set,
                      #I guess the steps per epoch could also be per data batch.
                     #steps_per_epoch= 5,
                     verbose=1,
                     epochs= 80)
