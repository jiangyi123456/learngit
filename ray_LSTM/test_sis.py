#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 14:10:41 2019

@author: shenlan
"""

from __future__ import division, print_function

import numpy as np

from keras.models import load_model
import pandas as pd
import tensorflow as tf
import os

#import os os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 


def adjust(train_x):
    newtrain_x = np.zeros([len(train_x),1,18],dtype=float)
   

    for i in range(len(train_x)):
        # newtrain_x.append(np.expand_dims(np.array(train_x[i]),axis=-1))
        newtrain_x[i] = np.array(train_x[i])
        

    return newtrain_x

#x_test =np.loadtxt("/home/shenlan/target/Labels/train_1/rayenergy_64_straight.txt")
#x_test = adjust(x_test)

train_data_dir = ('/media/lan/Net/rayshooting/target/train_1/CNN/train_rnn_cnn_data/pcmr1')

test_data_filename = 'data.txt'
res_data_filename='res.txt'

data_paths = []
res_paths = []

for case in os.listdir(train_data_dir):
    data_paths.append(os.path.join(train_data_dir,case,test_data_filename))
    res_paths.append(os.path.join(train_data_dir,case,res_data_filename))
    
for n in range(len(data_paths)):
    x_test=np.loadtxt(data_paths[n])
    x_test = adjust(x_test)  
    
    model=load_model('/media/lan/Net/rayshooting/target/Code/weights.best.hdf5')  

    np.set_printoptions(precision=4, threshold=10, edgeitems=5, linewidth=75, suppress=True)

    value = model.predict(x_test,batch_size=64)
# count = 0
    for i in range(len(value)):
        for j in range(len(value[i])):
            
            if(abs(value[i][j])<0.5):
                
                value[i][j]=0
            else:
                value[i][j]=1
   
    np.savetxt(res_paths[n],value)
       



#print('test1 value:',value)
# print(count)
# print(len(value))
# print(count/len(value))


