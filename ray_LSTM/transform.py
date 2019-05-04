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
import os
import keras
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Merge, LSTM, Dense
from keras.callbacks import ModelCheckpoint

model=load_model('/home/jy/programe/ray_LSTM/weights.best.hdf5')
print('   ')
