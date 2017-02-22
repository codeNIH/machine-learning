from __future__ import print_function
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution3D, MaxPooling3D
from keras.optimizers import SGD

model = Sequential()
# input: 100x100 images with 3 channels -> (3, 100, 100) tensors.
# this applies 32 convolution filters of size 3x3 each.
model.add(Convolution3D(32, 3, 3, 3, border_mode='valid', input_shape=(None, 20, 50, 50)))
model.add(Activation('relu'))
model.add(Convolution3D(32, 3, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Convolution3D(64, 3, 3, 3, border_mode='valid'))
model.add(Activation('relu'))
model.add(Convolution3D(64, 3, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
# Note: Keras does automatic shape inference.
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(10))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

X_train = np.load('../images.npy')
Y_train = np.load('../labels.npy')

model.fit(X_train, Y_train, batch_size=2, nb_epoch=1)
