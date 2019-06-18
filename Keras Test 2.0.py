#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 10:10:26 2018
keras test
@author: stevenchen
"""

import keras
#classification model
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
from keras.callbacks import Callback
import matplotlib.pyplot as plt
from keras import optimizers




class PrintAndSaveWeights(Callback):
    
    def on_epoch_end(self, batch, logs={}):
        """
        At the end of every epoch, save and print our slope and intercept weights
        """
        ## Get the current weights
        current_w = self.model.get_weights()
            
        ## Print them after each epoch
        print (current_w)

## Initialize our callback function for use in the model later
print_save_weights = PrintAndSaveWeights()


N = 5
Step =20
thetax = np.arange(0,N * np.pi,np.pi/Step)
r1 = 10*(1+thetax)
x1 = r1 * np.cos(thetax)
y1 = r1 * np.sin(thetax)
x1 = x1.reshape(-1,1)
y1 = y1.reshape(-1,1)
data1 = np.hstack([x1,y1])
# plt.scatter(data1[:, 0], data1[:, 1])

x2 = r1 * np.cos(10+thetax)
y2 = r1 * np.sin(10+thetax)
x2 = x2.reshape(-1,1)
y2 = y2.reshape(-1,1)
data2 = np.hstack([x2,y2])
# plt.scatter(data2[:,0], data2[:,1])


data = np.vstack([data1,data2])
# plt.scatter(data[:,0],data[:,1])

label = list(np.ones(N*Step))+list(0*np.ones(N*Step)) 
label = np.array(label)

def plot_decision_boundary(X, y, model, steps=1000, cmap='Paired'):
    """
    Function to plot the decision boundary and data points of a model.
    Data points are colored based on their actual label.
    """
    cmap = plt.get_cmap(cmap)
    
    # Define region of interest by data limits
    xmin, xmax = X[:,0].min() - 1, X[:,0].max() + 1
    ymin, ymax = X[:,1].min() - 1, X[:,1].max() + 1
    steps = 1000
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)

    # Make predictions across region of interest
    labels = model.predict(np.c_[xx.ravel(), yy.ravel()])
#    labels = labels > 0.5

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)
    
    return fig, ax

def logReg(X,Y,bs,ep):
    model = Sequential() # intilization, must have
    
    keras.optimizers.RMSprop(lr=0.01, epsilon=None, decay=0.0)

    model = Sequential([ 
            Dense(10, activation='relu', input_shape=(2,),kernel_initializer='random_uniform',
                bias_initializer='random_uniform')])
    
    model.add(Dense(50, activation='relu',kernel_initializer='random_uniform',
                bias_initializer='random_uniform'))
    
    model.add(Dense(60, activation='relu',kernel_initializer='random_uniform',
                bias_initializer='random_uniform'))
    
    model.add(Dense(30, activation='relu',kernel_initializer='random_uniform',
                bias_initializer='random_uniform'))

    model.add(Dense(1, activation='sigmoid',kernel_initializer='random_uniform',
                bias_initializer='random_uniform'))
    model.compile(loss='binary_crossentropy', optimizer='rmsprop',metrics=['accuracy'])
    
    model.fit(x=X,y=Y,batch_size=bs,epochs=ep,callbacks=[print_save_weights])
    
    return model


#logReg(data, label, 20, 3000)
plot_decision_boundary(data, label, logReg(data, label, 150, 12000))
