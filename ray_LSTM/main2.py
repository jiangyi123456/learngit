# -*- coding: UTF-8 -*-
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
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D,BatchNormalization
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array, load_img
from PIL import Image
import matplotlib.pyplot as plt
from itertools import chain

import plistlib
import numpy as np
import os
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense
from keras.callbacks import ModelCheckpoint

def argument_data(train_x,train_x1,train_y):  # train_x is traindata and train_y is label
    newtrain_x = np.zeros([len(train_x),1,len(train_x[0])],dtype=float)
    newtrain_y = np.zeros([len(train_x),len(train_x[0])],dtype=float)
    newtrain_x1=np.zeros([len(train_x),1,len(train_x[0])],dtype=float)
    for i in range(len(train_x)):
        # newtrain_x.append(np.expand_dims(np.array(train_x[i]),axis=-1))
        newtrain_x[i] = np.array(train_x[i])
        newtrain_y[i] = np.array(train_y[i])
        newtrain_x1[i] = np.array(train_x1[i])

#    for i in range(len(train_x)):
 #       temp = reversed(train_x[i])
  #      temp1 =reversed(train_y[i])
   #     temp2=reversed(train_x1[i])
    #    # newtrain_x.append(np.expand_dims(np.array(list(temp)),axis=-1))
     #   newtrain_x[i+len(train_x)] = np.array(list(temp))
      #  newtrain_y[i+len(train_x)] = np.array(list(temp1))
       # newtrain_x1[i + len(train_x)] = np.array(list(temp2))
        # print(newtrain_x)

    return newtrain_x,newtrain_x1,newtrain_y

def rotate(train_x,train_y):
    newtrain_x = np.zeros([len(train_x)*2,1,18],dtype=float)
    newtrain_y = np.zeros([len(train_x)*2,18],dtype=float)

    for num in range(len(train_x)):
        j = int(x_train.shape[1]/4*3)
        for i in range(0,int(x_train.shape[1]/4)):
            # print(num,i,num,j)
            # print(train_x.shape)
            newtrain_x[num][0][i] = np.array(int(train_x[num][j]))
            newtrain_y[i] = np.array(train_y[i])
            j = j+1

        j = 0
        for i in range(int(x_train.shape[1]/4),int(x_train.shape[1]/4*2)):
            newtrain_x[num][0][i] = np.array(int(train_x[num][j]))
            newtrain_y[i] = np.array(train_y[i])
            j = j+1

        j = int(x_train.shape[1]/4)
        for i in range(int(x_train.shape[1]/4*2),x_train.shape[1]):
            newtrain_x[num][0][i] = np.array(int(train_x[num][j]))
            newtrain_y[i] = np.array(train_y[i])
            j = j+1


        j = 0
        for i in range(len(train_x),2*len(train_x)):
            newtrain_x[i] = np.array(train_x[j])
            newtrain_y[i] = np.array(train_y[j])
            j = j + 1

    return newtrain_x,newtrain_y

def adjust(train_x,train_x1,train_y):
    newtrain_x = np.zeros([len(train_x),1,len(train_x[0])],dtype=float)
    newtrain_y = np.zeros([len(train_y),len(train_x[0])],dtype=float)
    newtrain_x1 = np.zeros([len(train_x), 1, len(train_x[0])], dtype=float)
    for i in range(len(train_x)):
        # newtrain_x.append(np.expand_dims(np.array(train_x[i]),axis=-1))
        newtrain_x[i] = np.array(train_x[i])
        newtrain_y[i] = np.array(train_y[i])
        newtrain_x1[i] = np.array(train_x1[i])
    return newtrain_x,newtrain_x1,newtrain_y


x_train = np.loadtxt("/home/jy/programepy/ray_LSTM/train_data/RayTdata.txt")#能量
x_train1= np.loadtxt("/home/jy/programepy/ray_LSTM/train_data/RayTdata1.txt")
y_train = np.loadtxt("/home/jy/programepy/ray_LSTM/train_data/rayTL.txt")#标签

x_train,x_train1 ,y_train = argument_data(x_train,x_train1,y_train)  # deal whit the data

x_test =np.loadtxt("/home/jy/programepy/ray_LSTM/train_data/Raytdata.txt")
x_test1 =np.loadtxt("/home/jy/programepy/ray_LSTM/train_data/Raytdata1.txt")
y_test =np.loadtxt("/home/jy/programepy/ray_LSTM/train_data/raytL.txt")

x_test,x_test1,y_test = adjust(x_test,x_test1,y_test)  # adapt the output data

#写一个LossHistory类，保存loss和acc
#loss和acc的变化可视化
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        # acc
        plt.plot(iters, self.accuracy[loss_type], 'r', label='train acc')
        # loss
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        if loss_type == 'epoch':
            # val_acc
            plt.plot(iters, self.val_acc[loss_type], 'b', label='val acc')
            # val_loss
            plt.plot(iters, self.val_loss[loss_type], 'k', label='val loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('acc-loss')
        plt.legend(loc="upper right")
        plt.show()

model_flag = 0
max_features=2000
save_weights_flag =1
model_name = '1558_3_256_20_'
################################################     定义模型
#model = Sequential()
#model.add(LSTM(100,input_shape=(x_train.shape[1],x_train.shape[2]))) #100输出维度
#model.add(Dense(18, activation='softmax')) # 全链接层18该层的输出维度激活函数
#model.summary()

encoder_a = Sequential()
encoder_a.add(LSTM(100, input_shape=(x_train.shape[1],x_train.shape[2])))

encoder_b = Sequential()
encoder_b.add(LSTM(100, input_shape=(x_train.shape[1],x_train.shape[2])))


decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b], mode='concat')) # output is 200
decoder.add(Dense(100, activation='relu'))

decoder.add(Dense(19,activation='relu'))

##############################################

history = LossHistory() # 实例化类
if save_weights_flag:
    adam = keras.optimizers.adam(lr=1e-5) # 学习率0.001
    decoder.compile(loss='msle',
                  optimizer='adam',
                  metrics=['accuracy']) #损失函数 平均绝对误差
    # checkpoint
    filepath='weights.best.hdf5'
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True,save_weights_only=False,
                                mode='auto')
    callbacks_list = [checkpoint]

    hist = decoder.fit([x_train,x_train1], y_train,validation_split=0.0,
              epochs=200,
              batch_size=64,callbacks=[history],validation_data=([x_test,x_test1],y_test))
     #batch_size = 32, callbacks = [history], validation_data = ([x_test, x_test], y_test))
    history.loss_plot('epoch')
    decoder.save(filepath)
else:
    decoder.load_weights('weights.best.hdf5')
    decoder.compile(loss='msle', optimizer='adam', metrics=['accuracy'])

value = decoder.predict([x_test,x_test1],batch_size=64)
value1=np.round(value)
value1=value-y_test
value1=sum(sum(value1))
totlenum=y_test.size
rate=value1/totlenum
print('rate:',rate)
# count = 0
for i in range(len(value)):
    for j in range(len(value[i])):
        if (abs(value[i][j]) < 0.5):
            value[i][j] = 0
        else:
            value[i][j] = 1

result = 0
num = 0
for i in range(len(value)):
    for j in range(len(value[i])):
        result = result + 1
        if (abs(value[i][j]) == abs(y_test[i][j])):
            num += 1

accracy1 = float(num / result)
#value=model.evaluate( [x_test,x_test], y_test, batch_size=32, verbose=1, sample_weight=None)
np.set_printoptions(precision=4, threshold=10, edgeitems=5, linewidth=75, suppress=True)
print('test1 value:',accracy1)
