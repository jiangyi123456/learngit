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
from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense,Dropout,Bidirectional,Concatenate
from keras.callbacks import ModelCheckpoint
'''
def argument_data(train_x,train_y):
    newtrain_x = np.zeros([len(train_x)*2,1,18],dtype=float)
    newtrain_y = np.zeros([len(train_x)*2,18],dtype=float)

    for i in range(len(train_x)):
        # newtrain_x.append(np.expand_dims(np.array(train_x[i]),axis=-1))
        newtrain_x[i] = np.array(train_x[i])
        newtrain_y[i] = np.array(train_y[i])

    for i in range(len(train_x)):
        temp = reversed(train_x[i])
        # newtrain_x.append(np.expand_dims(np.array(list(temp)),axis=-1))
        newtrain_x[i+len(train_x)] = np.array(list(temp))
        newtrain_y[i+len(train_x)] = np.array(train_y[i])


        # print(newtrain_x)

    return newtrain_x,newtrain_y
'''
def argument_data(train_x,train_x1,train_y):  # train_x is traindata and train_y is label
    newtrain_x = np.zeros([len(train_x),1,len(train_x[0])],dtype=float)
    newtrain_y = np.zeros([len(train_x),len(train_x[0])],dtype=float)
    newtrain_x1=np.zeros([len(train_x),1,len(train_x[0])],dtype=float)
    for i in range(len(train_x)):
        # newtrain_x.append(np.expand_dims(np.array(train_x[i]),axis=-1))
        newtrain_x[i] = np.array(train_x[i])
        newtrain_y[i] = np.array(train_y[i])
        newtrain_x1[i] = np.array(train_x1[i])
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

def adjust(train_x,train_y):
    newtrain_x = np.zeros([len(train_x),1,18],dtype=float)
    newtrain_y = np.zeros([len(train_y),18],dtype=float)

    for i in range(len(train_x)):
        # newtrain_x.append(np.expand_dims(np.array(train_x[i]),axis=-1))
        newtrain_x[i] = np.array(train_x[i])
        newtrain_y[i] = np.array(train_y[i])

    return newtrain_x,newtrain_y


x_train = np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/RayTdata.txt")#能量
x_train1= np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/RayTdata1.txt")
y_train = np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/rayTL.txt")#标签

x_train,x_train1 ,y_train = argument_data(x_train,x_train1,y_train)  # deal whit the data

x_test =np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/Raytdata.txt")
x_test1 =np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/Raytdata1.txt")
y_test =np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/raytL.txt")

x_test,x_test1,y_test = argument_data(x_test,x_test1,y_test)  # adapt the output data

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


class Save(keras.callbacks.Callback):
    def __init__(self):
        self.max_acc = 0.0

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        self.val_acc = logs["val_acc"]
        if epoch != 0:
            if self.val_acc > self.max_acc and self.val_acc > 0.8:
                decoder.save("kears_model_" + str(epoch) + "_acc=" + str(self.val_acc) + ".h5")
                self.max_acc = self.val_acc


save_function = Save()


model_flag = 0
max_features=2000
save_weights_flag = 1
model_name = '1558_3_256_20_'
################################################     定义模型
#model = Sequential()
#model.add(LSTM(100,input_shape=(x_train.shape[1],x_train.shape[2]))) #100输出维度
#model.add(Dense(18, activation='softmax')) # 全链接层18该层的输出维度激活函数
#model.summary()
encoder_a = Sequential()

encoder_a.add(Bidirectional(LSTM(256 ,activation='softsign'),input_shape=(x_train.shape[1],x_train.shape[2]))) #100  , ,input_shape=(x_train.shape[1],x_train.shape[2])  return_sequences=True),
#encoder_a.add(Bidirectional(LSTM(128)) ) # ,return_sequences=True    ,activation='softsign'

encoder_b = Sequential()

encoder_b.add(Bidirectional(LSTM(256,activation='softsign' ),input_shape=(x_train.shape[1],x_train.shape[2]))) #100  ,activation='softsign' ,input_shape=(x_train.shape[1],x_train.shape[2])   ,return_sequences=True)
#encoder_b.add(Bidirectional(LSTM(128)))

decoder = Sequential()
decoder.add(Merge([encoder_a, encoder_b], mode='concat'))   #

decoder.add(Dense(128,activation='relu')) # ,activation='softplus'
decoder.add(Dense(19,activation='sigmoid',kernel_regularizer=regularizers.l2(0.025)))   #　　,activity_regularizer=regularizers.l1(0.001)  ,kernel_regularizer=regularizers.l2(0.002)   sigmoid

##############################################

history = LossHistory() # 实例化类
if save_weights_flag:
    #adam = keras.optimizers.adam(lr=1e-2,decay=1e-6) # 学习率0.001
    adam=keras.optimizers.Adagrad(lr=0.01,  decay=0.0)
    #adam=Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
    decoder.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy']) #损失函数 平均绝对误差    binary_crossentropy

    # checkpoint
    filepath='weights.best.hdf5'
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, save_weights_only=True,
                                mode='auto')
    callbacks_list = [checkpoint]

    hist = decoder.fit([x_train,x_train1], y_train,validation_split=0.3,
              epochs=600,
              batch_size=128,callbacks=[history],validation_data=([x_test,x_test1],y_test))

     #batch_size = 32, callbacks = [history], validation_data = ([x_test, x_test], y_test))   validation_split=0.2,

    history.loss_plot('epoch')
    decoder.save(filepath)
else:
    decoder.load_weights('weights.best.hdf5')
    adam = Adagrad(lr=0.01, decay=0.0)
    decoder.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])
value = decoder.predict([x_test, x_test1], batch_size=64)
value1 = np.round (value)   #rint
#np.savetxt("result.txt", value1)
text=value1-y_test
num = str(text.tolist()).count("0")
totalnum=x_test.size
rate=(num-totalnum)/totalnum
print(rate)
numb=0
for i in range(len(text)):
    for j in range(1,len(text[i])-1):
        if (abs(text[i][j])==1)and(abs(text[i][j+1])==0 and abs(text[i][j-1]==0)):
            numb += 1

accracy1 = float(numb / totalnum)
print('rate:',accracy1)
score = decoder.evaluate([x_test,x_test],y_test,batch_size=32)
print('test1:',score)


#  Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

