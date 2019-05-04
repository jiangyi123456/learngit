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
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, UpSampling2D, Dropout, Cropping2D
from keras.optimizers import *
from keras.preprocessing.image import ImageDataGenerator, array_to_img,img_to_array, load_img
from PIL import Image
import matplotlib.pyplot as plt
from itertools import chain

import plistlib
import numpy as np
import random
import os
import keras
from keras.utils import to_categorical
from keras.layers import LSTM
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
from keras.callbacks import ModelCheckpoint


################################################################################################
#####################     before design the program, you can make the label of data by below
# PATH = 'E:/CharlesGuo/PycharmProjects/raylength/positive/'
# frulyfly1_xyloc_raylen = np.loadtxt(PATH+"frulyfly1_xyloc_raylen.txt")
# # print(frulyfly1_xyloc_raylen)
# frulyfly1_xyloc_raylen_row = len(frulyfly1_xyloc_raylen)
# # print(frulyfly1_xyloc_raylen_row)
# frulyfly1_xyloc_raylen_label = np.ones([frulyfly1_xyloc_raylen_row,1])
# # print(frulyfly1_xyloc_raylen_label)
# #####     see whether the file exist
# existflag = os.path.exists(PATH+"frulyfly1_xyloc_raylen_label.txt")
# if(existflag):
#     print("目录有重名文件")
#     os.remove(PATH+"frulyfly1_xyloc_raylen_label.txt")
#     print('已删除，移除后目录下有文件：%s' % os.listdir(PATH))
#
# np.savetxt(PATH+"frulyfly1_xyloc_raylen_label.txt",frulyfly1_xyloc_raylen_label,fmt='%d')     #将数组中数据写入到data.txt文件
# print('已生成frulyfly1_xyloc_raylen_label')
#
# no_cluster_thin_frulyfly1_xyloc_raylen = np.loadtxt(PATH+"no_cluster_thin_frulyfly1_xyloc_raylen.txt")
# # print(frulyfly1_xyloc_raylen)
# no_cluster_thin_frulyfly1_xyloc_raylen_row = len(no_cluster_thin_frulyfly1_xyloc_raylen)
# # print(frulyfly1_xyloc_raylen_row)
# no_cluster_thin_frulyfly1_xyloc_raylen_label = np.ones([no_cluster_thin_frulyfly1_xyloc_raylen_row,1])
# # print(frulyfly1_xyloc_raylen_label)
# #####     see whether the file exist
# existflag = os.path.exists(PATH+"frulyfly1_xyloc_raylen_label.txt")
# if(existflag):
#     print("目录有重名文件")
#     os.remove(PATH+"frulyfly1_xyloc_raylen_label.txt")
#     print('已删除，移除后目录下有文件：%s' % os.listdir(PATH))
#
# np.savetxt(PATH+"frulyfly1_xyloc_raylen_label.txt",no_cluster_thin_frulyfly1_xyloc_raylen_label,fmt='%d')     #将数组中数据写入到data.txt文件
# print('已生成frulyfly1_xyloc_raylen_label')

####################     make the label of the thin branch points of frulyfly1
# filename = PATH+'no_cluster_thin_frulyfly1_xyloc_raylen.txt'
# true_ind = [190,221,194,254,275,262,199,196,292,283,243,234,230,222,225,183,132,181,256,262,220,232]
# data = np.zeros((1,313))
# data = data.reshape((-1,1))
# for i in range(1,len(true_ind)):
#     data[true_ind[i]-1] = 1
# np.savetxt(PATH+"no_cluster_thin_frulyfly1_xyloc_raylen_label.txt",data,fmt='%d')     #将数组中数据写入到data.txt文件


# #####################     define the train data
# train_data = np.loadtxt(PATH+'new_no_cluster_thin_frulyfly1_xyloc_raylen.txt')
#
# train_data_label = np.loadtxt(PATH+'new_no_cluster_thin_frulyfly1_xyloc_raylen_label.txt')
# train_data_label = train_data_label.reshape(-1,1)
# # print(train_data)
# # print(train_data_label)
#
# #####################     define the test data
# frulyfly1_xyloc_raylen = np.loadtxt(PATH+'frulyfly1_xyloc_raylen.txt')
# frulyfly1_xyloc_raylen_label = np.loadtxt(PATH+'frulyfly1_xyloc_raylen_label.txt')
# frulyfly1_xyloc_raylen_label = frulyfly1_xyloc_raylen_label.reshape(-1,1)
# # print(frulyfly1_xyloc_raylen)
# # print(frulyfly1_xyloc_raylen_label)
# frulyfly2_xyloc_raylen = np.loadtxt('E:/CharlesGuo/PycharmProjects/raylength/negative/'+'frulyfly2_xyloc_raylen.txt')
# frulyfly2_xyloc_raylen_label = np.loadtxt('E:/CharlesGuo/PycharmProjects/raylength/negative/'+'frulyfly2_xyloc_raylen_label.txt')
# frulyfly2_xyloc_raylen_label = frulyfly2_xyloc_raylen_label.reshape(-1,1)
# def train():
#     train_x = train_data.tolist()
#     train_y = train_data_label.tolist()
#     return train_x,train_y
#
# def test():
#     return(frulyfly1_xyloc_raylen,frulyfly1_xyloc_raylen_label)
'''
###################-------打乱数据：----------
#'数据：', data
#'标签：', y
################################################################################################
'''
rand_data = np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/RayTraindata_op1.txt")
rand_label = np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/RayTrainlabel_op1.txt")

seedint = random.randint(1, len(rand_label))     # 返回闭区间 [a, b] 范围内的整数值
# print(seedint)

np.random.seed(seedint)
np.random.shuffle(rand_data)
np.random.seed(seedint)
np.random.shuffle(rand_label)


def train_data():
    
    return(rand_data)

def train_label():
    return(rand_label)
#test_row = thin_frulyfly1_label_row
# print(test_data())
# print(x1)


def argument_data(train_x,train_y):
    newtrain_x = np.zeros([len(train_x)*2,1,18],dtype=float)
    newtrain_y = np.zeros([len(train_x)*2,18],dtype=float)

    for i in range(len(train_x)):
        # newtrain_x.append(np.expand_dims(np.array(train_x[i]),axis=-1))
        newtrain_x[i] = np.array(train_x[i])
        newtrain_y[i] = np.array(train_y[i])

    for i in range(len(train_x)):
        temp = reversed(train_x[i])
        temp1= reversed(train_y[i])
        # newtrain_x.append(np.expand_dims(np.array(list(temp)),axis=-1))
        newtrain_x[i+len(train_x)] = np.array(list(temp))
        newtrain_y[i+len(train_x)] = np.array(list(temp1))


        # print(newtrain_x)

    return newtrain_x,newtrain_y

def adjust(train_x,train_y):
    newtrain_x = np.zeros([len(train_x),1,18],dtype=float)
    newtrain_y = np.zeros([len(train_y),18],dtype=float)

    for i in range(len(train_x)):
        # newtrain_x.append(np.expand_dims(np.array(train_x[i]),axis=-1))
        newtrain_x[i] = np.array(train_x[i])
        newtrain_y[i] = np.array(train_y[i])

    return newtrain_x,newtrain_y
'''
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
'''


x_train = train_data()
y_train = train_label()
# print(len(x_train))
# x_train ,y_train = rotate(x_train ,y_train )
# print(len(x_train))
# print(y_train.shape)
# print(len(y_train))

x_train ,y_train = argument_data(x_train,y_train)
# print(len(x_train))


x_test =np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/raytestdata_op1.txt")
y_test =np.loadtxt("/home/jy/programepy/ray_LSTM/train_1/raytestlabel_op1.txt")
x_test,y_test = adjust(x_test,y_test)


#写一个LossHistory类，保存loss和acc
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
save_weights_flag = 1
model_name = '1558_3_256_20_'
################################################     定义模型
model = Sequential()
#model.add(LSTM(200,input_shape=(x_train.shape[1],x_train.shape[2]),return_sequences=True)) 
model.add(LSTM(200,input_shape=(x_train.shape[1],x_train.shape[2])))
#model.add(Bidirectional(LSTM(200,input_shape=(x_train.shape[1],x_train.shape[2]))))        
#model.add(TimeDistributed(Dense(1,  activation='sigmoid')))

#model.add(Dense(128, activation='relu'))
# model.add(Dense(256,  input_shape=(1,18),activation='relu'))
# model.add(Dropout(0.5))
#model.add(Flatten())
#model.add(Dense(1))
model.add(Dense(18))

model.summary()
##############################################

history = LossHistory()
if save_weights_flag:
    adam = keras.optimizers.adam(lr=1e-6)
    model.compile(loss='mse',
                  optimizer='adam',
                  metrics=['accuracy'])
    # checkpoint
    filepath='weights.best.hdf5'
    # 有一次提升, 则覆盖一次.
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True,
                                mode='max')
    callbacks_list = [checkpoint]
    #y_train = keras.utils.to_categorical(y_train, num_classes=None)
    #x_train=keras.utils.to_categorical(x_train, num_classes=None)
    #y_test = keras.utils.to_categorical(y_test, num_classes=None)
   # x_test= keras.utils.to_categorical(x_test, num_classes=None)
    hist = model.fit(x_train, y_train,validation_split=0.2,
              epochs=100,
              batch_size=64,callbacks=[history],validation_data=(x_test,y_test))
    history.loss_plot('epoch')
else:
    model.load_weights('weights.best.hdf5')
    model.compile(loss='mse ', optimizer='adam', metrics=['accuracy'])
model.save("weights.best.hdf5")
    #model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# if model_flag:
#     model.save(model_name)
# else:
#     model = load_model(model_name)


########## adam    rmsprop       categorical_crossentropy      binary_crossentropy





###############################   yanzheng
np.set_printoptions(precision=4, threshold=10, edgeitems=5, linewidth=75, suppress=True)

value = model.predict(x_test,batch_size=64)
# count = 0
value1=np.round(value)
value1=value-y_test
numzero=str(value1.tolist()).count("0")
totlenum=y_test.size
rate=(numzero-totlenum)/totlenum

for i in range(len(value)):
    for j in range(len(value[i])):
        if(abs(value[i][j])<0.5):
         value[i][j]=0
        else:
         value[i][j]=1

result=0
num=0
for i in range(len(value)):
    for j in range(len(value[i])):
        result=result+1
        if(abs(value[i][j])==abs(y_test[i][j])):
         num+=1
    
accracy1=float(num/result)

print('test1 value:',value)
# print(count)
# print(len(value))
# print(count/len(value))

score = model.evaluate(x_test,y_test,batch_size=64)
print('test1:',score)
print('test1_acc:',accracy1)


#score2 = model.evaluate(x_test2,y_test2,batch_size=20)
#print('test2:',score2)

#score3 = model.evaluate(x_test3,y_test3,batch_size=20)
#print('test3:',score3)