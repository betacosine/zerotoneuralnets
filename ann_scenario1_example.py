# -*- coding: utf-8 -*-
"""
Created on Tue May 12 09:07:37 2020

@author: BetaCosine
"""

'''
Chapter 6:
This program builds a multi-layer feedforward neural network 
using the Keras library
The program leverages the preprocessing steps of the program 
data_science_process_chapters1_4.py

'''
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix

path = 'C:/Users/BetaCosine/Google Drive/BetaCosine/Ebooks/'

model = Sequential()
model.add(Dense(16, input_dim=21, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', 
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=200, batch_size=25)

#summarize the model

print(model.summary())

#check the model against the test data

_, accuracy_train = model.evaluate(x_train, y_train)
_, accuracy_test = model.evaluate(x_test, y_test)

pred_nn = model.predict_classes(x_test)

conf_mat_nn = confusion_matrix(y_test, pred_nn)
tn_nn, fp_nn, fn_nn, tp_nn = conf_mat_nn.ravel()
ppv_nn = tp_nn/(tp_nn+fp_nn)
print(ppv_nn)

