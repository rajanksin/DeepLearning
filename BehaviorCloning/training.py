#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 29 10:50:25 2017

@author: srajanku
"""

import csv
import cv2
import numpy as np

import keras as kr
from keras.layers import Input, Flatten, Dense, Lambda, Dropout, Activation, Cropping2D
from keras.models import Model, Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

lines = []
images=[]
measurements=[]

steering_correction = 0.2
CENTER = 0.0
LEFT   = steering_correction
RIGHT  = -1* steering_correction

def addImages(sourcePath, correction):
    filename = sourcePath.split('/')[-1]
    currPath = './mydata/Track01/data-03-clockwise/IMG/' + filename
#    print(currPath)
    image=cv2.imread(currPath)
    images.append(image)
    measure = float(line[3]) + correction
    measurements.append(measure)
#    flip image
    image_flip = cv2.flip(image, 1)
    images.append(image_flip)
    measurements.append(measure * -1) 


with open('./myData/Track01/data-03-clockwise/driving_log.csv') as csvfile:
    reader= csv.reader(csvfile)
    next(reader,None)
    for line in reader:
        lines.append(line)



for line in lines:
    
#    center
    addImages(line[0], CENTER)
#    left
    addImages(line[1], LEFT)
#    right
    addImages(line[2], RIGHT)
    

X_train = np.array(images)
y_train = np.array(measurements)

model = Sequential()
model.add(Lambda(lambda x : x/255 -0.5, input_shape= X_train.shape[1:]))

model.add(Cropping2D(cropping=((70,25), (0,0))))

model.add(Convolution2D(25, 5, 5, subsample=(2,2),activation="relu"))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))

model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation="relu"))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))

model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation="relu"))
#model.add(MaxPooling2D((2, 2)))
#model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(Dropout(0.5))

model.add(Convolution2D(64, 3, 3, activation="relu"))
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(120))
#model.add(Activation('relu'))

model.add(Dense(84))
#model.add(Activation('relu'))

#model.add(Dense(1))
model.add(Dense(16))
#model.add(Activation('relu'))

#model.add(Flatten())
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, batch_size=256, validation_split=0.2 , shuffle=True, nb_epoch=2)

model.save('./myData/Track01/data-03-clockwise/model_cropping.h5')