from __future__ import division, print_function
from numpy.core.multiarray import ndarray
from scipy.misc import imresize
from keras.applications import vgg16
import matplotlib.pyplot as plt
from keras.layers.core import Activation, Dense, Dropout, Lambda,Flatten
from keras.layers.merge import Concatenate
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import itertools
from keras.models import *
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array, load_img
from PIL import Image
import matplotlib.pyplot as plt
from itertools import chain

import plistlib
import numpy as np
import os
import keras
from keras import regularizers
from keras.models import Model
from keras.layers import Merge, LSTM, Dense,Dropout,Bidirectional
from keras.callbacks import ModelCheckpoint

def argument_data(train_x,train_x1):  # train_x is traindata and train_y is label
    newtrain_x = np.zeros([len(train_x),1,len(train_x[0])],dtype=float)
    newtrain1_x = np.zeros([len(train_x),1,len(train_x[0])],dtype=float)

    for i in range(len(train_x)):
        # newtrain_x.append(np.expand_dims(np.array(train_x[i]),axis=-1))
        newtrain_x[i] = np.array(train_x[i])
        newtrain1_x[i] = np.array(train_x1[i])

    return newtrain_x,newtrain1_x



x_train = np.loadtxt("/home/jy/programepy/ray_LSTM/train_data/RayTraindata.txt")#能量
x_train1= np.loadtxt("/home/jy/programepy/ray_LSTM/train_data/RayTraindata1.txt")


x_text,x_text1 = argument_data(x_train,x_train1)  # deal whit the data

################################################     定义模型
encoder_a = Sequential()
encoder_a.add(Bidirectional(LSTM(256 ,activation='softsign'),input_shape=(1,19))) #100  , ,input_shape=(x_train.shape[1],x_train.shape[2])  return_sequences=True),
encoder_b = Sequential()
encoder_b.add(Bidirectional(LSTM(256,activation='softsign' ),input_shape=(1,19))) #100  ,activation='softsign' ,input_shape=(x_train.shape[1],x_train.shape[2])   ,return_sequences=True)

decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b], mode='concat'))   #
decoder.add(Dense(128,activation='relu'))
decoder.add(Dense(19,activation='sigmoid',kernel_regularizer=regularizers.l2(0.015)))
##############################################

decoder.load_weights('weights.best.hdf5')
adam = Adagrad(lr=0.01, decay=0.0)
decoder.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
value = decoder.predict([x_text, x_text1], batch_size=64)
value1 = np.round (value)   #rint
np.savetxt("/media/jy/ubuntu/matlab2018b/program/2D/LSTMlearning/Labels_old/YPred.txt", value1)

