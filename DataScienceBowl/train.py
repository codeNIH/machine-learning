from __future__ import print_function
import os
import sys
import numpy as np
from keras.layers import merge, Input
from keras.layers.convolutional import Convolution3D, ZeroPadding3D, MaxPooling3D, AveragePooling3D
from keras.layers.core import Dense, Activation, Flatten, Dropout, Merge
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.optimizers import Adam, Nadam, SGD
import utils
import tensorflow as tf
from keras import backend as K
from keras.models import Model
from keras.layers import Input, merge
from keras.layers.core import Lambda

model = Sequential()
