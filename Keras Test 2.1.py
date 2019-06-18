#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 28 13:01:07 2018

@author: stevenchen
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from keras import optimizers
from keras.callbacks import Callback
from sklearn import datasets, linear_model





class PrintAndSaveWeights(Callback):
    
    def on_epoch_end(self, batch, logs={}):
        """
        At the end of every epoch, save and print our slope and intercept weights
        """
        ## Get the current weights
        current_w = self.model.get_weights()
            
        ## Print them after each epoch
        print (current_w)
        
print_save_weights = PrintAndSaveWeights()


#y_binary = to_categorical(y_int)



x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y=[1, 2, 3]

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
   # labels = labels > 0.5

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
    
    model = Sequential([ 
            Dense(3, activation='softmax', input_shape=(3,),kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        ])
    
    model.add(Dense(1, activation='softmax', input_shape=(3,),kernel_initializer='random_uniform',
                bias_initializer='random_uniform'))
    
    
   
    model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])
    
    model.fit(x=X,y=Y,batch_size=bs,epochs=ep,callbacks=[print_save_weights])
    
    return model


#logReg(x, y, 3, 100)
plot_decision_boundary(x, y, logReg(x, y, 3, 100))