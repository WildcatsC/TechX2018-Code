
# coding: utf-8

# In[1]:


import keras

from sklearn import datasets, linear_model

from keras.models import Sequential
from keras.layers import Dense
import numpy as np
from keras.utils import to_categorical


# In[192]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
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
    labels = labels > 0.5

    # Plot decision boundary in region of interest
    z = labels.reshape(xx.shape)
    
    fig, ax = plt.subplots()
    ax.contourf(xx, yy, z, cmap=cmap, alpha=0.5)

    # Get predicted labels on training data and plot
    train_labels = model.predict(X)
    ax.scatter(X[:,0], X[:,1], c=y, cmap=cmap, lw=0)
    
    return fig, ax


# In[186]:


import random
x=[[random.random(),random.random()] for i in range(100)] + [[random.random()+100,random.random()+100] for i in range(100)]
y = [1 for i in range(100)]+[0 for i in range(100)]

y_binary = to_categorical(y)


# In[189]:


np.array(x).shape


# In[191]:


for i in range(10):
    x[i][0]=10
    x[i][1]=20
    x[i+10][0]=20
    x[i+10][1]=10


# In[193]:


from keras.callbacks import Callback

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


# In[194]:


def logReg(X,Y,bs,ep):
    model = Sequential() # intilization, must have
    
    model = Sequential([ 
            Dense(1, activation='sigmoid', input_shape=(2,),kernel_initializer='random_uniform',
                bias_initializer='random_uniform')
        ])
    
    
    model.compile(loss='binary_crossentropy', optimizer='sgd',metrics=['accuracy'])
    
    model.fit(x=X,y=Y,batch_size=bs,epochs=ep,callbacks=[print_save_weights])
    return model


# In[195]:


m = logReg(np.array(x).reshape(200,2),y,25,500)


# In[140]:


plot_decision_boundary(np.array(x),y,m)


# In[125]:


sampleModel = m


# In[196]:


def linReg(X,Y,bs=15,ep=500):
    model = Sequential()
    model = Sequential([
            Dense(1, activation='linear', input_shape=(1,),kernel_initializer='glorot_uniform',
                bias_initializer='glorot_uniform')
        ])
    sgd = keras.optimizers.SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='mean_squared_error', optimizer=sgd,metrics=['accuracy'])
    model.fit(x=X,y=Y,batch_size=bs,epochs=ep,callbacks=[print_save_weights])
    return model


# In[200]:


x=np.arange(100)
y=np.arange(100)
for i in range(len(y)):
    y[i] = 3*x[i]+4 - random.randint(-4,4)*4


# In[202]:


m = linReg(x,y)


# In[201]:


plt.plot(x,y)


# In[203]:


plt.plot(x,m.predict(x),y)


# In[185]:


# %matplotlib inline
# import matplotlib.pyplot as plt
# import numpy as np

N = 5
Step =20
thetax = np.arange(0,N * np.pi,np.pi/Step)
r1 =10*(1+thetax)
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


#


# In[177]:


