# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 04:42:18 2017

@author: staslist
"""

import numpy as np
import mltools as ml
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

np.random.seed(0)

# Data Loading
X = np.genfromtxt('data/X_train.txt', delimiter=None)
Y = np.genfromtxt('data/Y_train.txt', delimiter=None)

# The test data
Xte = np.genfromtxt('data/X_test.txt', delimiter=None)

Xtr, Xva, Ytr, Yva = ml.splitData(X, Y)
Xtr, Ytr = ml.shuffleData(Xtr, Ytr)

# Taking a subsample of the data so that trains faster.
Xt, Yt = Xtr[:1000], Ytr[:1000] 

XtS, params = ml.rescale(Xt)
XvS, _ = ml.rescale(Xva, params)

model = Sequential()
model.add(Dense(14, input_dim=14, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(XtS, Yt, epochs=150, batch_size=10)
# evaluate the model
scores = model.evaluate(XtS, Yt)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# used the following tutorial to begin using keras library
# https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/