#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:59:15 2017

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
from keras.models import load_model

model = load_model('./myData/Track01/data-03-clockwise/model_cropping.h5')
model.summary()

lines = []
images=[]
measurements=[]

steering_correction = 0.2
CENTER = 0.0
LEFT   = steering_correction
RIGHT  = -1* steering_correction

def addImages(sourcePath, correction):
    filename = sourcePath.split('/')[-1]
    currPath = './mydata/Track02/Data-01/IMG/' + filename
#    print(currPath)
    image=cv2.imread(currPath)
    images.append(image)
    measure = float(line[3]) + correction
    measurements.append(measure)
#    flip image
    image_flip = cv2.flip(image, 1)
    images.append(image_flip)
    measurements.append(measure * -1) 


with open('./mydata/Track02/Data-01/driving_log.csv') as csvfile:
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

model.compile(loss='mse', optimizer='adam')

for layer in model.layers[:-4]:
    layer.trainable = False

model.summary()

model.fit(X_train, y_train, batch_size=256, validation_split=0.2 , shuffle=True, nb_epoch=2)

#set all layers to trainable , so that saving saves complete model.
# not doing this saves only layers which were trained.
for layer in model.layers[:-4]:
    layer.trainable = True
model.save('./mydata/Track02/model_cloned.h5')