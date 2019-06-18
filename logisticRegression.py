
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def logistic_regression(X, t):
    initial_w = np.asmatrix([-4, -2])
    num_iter = 1000
    learning_rate = 0.01
    
    w = gradient_decent(X, t, initial_w, num_iter, learning_rate)
    
    return w

def gradient_decent(X, t, w, ni, lr):
    for i in range(ni):
        dw = delta_w(w, X, t, lr)
        w = w - dw
    return w

def delta_w(w_k, x, t, learning_rate):
    return learning_rate * gradient(w_k, x, t)

def gradient(w, x, t):
    return (nn(x, w) - t).T * x

def nn(x, w):
    return logistic(x.dot(w.T))

def logistic(z):
    return 1 / (1 + np.exp(-z))

