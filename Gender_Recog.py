
# coding: utf-8

# In[1]:


###GenderRecogModel###

#GenderRecog
#July 29, 2018
#MaLinXuan


# In[2]:


#Dependencies

import numpy as np
#from os import listdir
#from cv2 import imread
#from random import randint
from keras.models import Sequential
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.layers import Dropout,Conv2D,MaxPooling2D,Flatten,Dense
from keras.callbacks import Callback


# In[3]:
class SaveModel(Callback):
    def on_epoch_end(self, batch, logs={}):
        if batch%1000==0:
            self.model.save('{}.h5'.format(batch))

## Initialize our callback function for use in the model later
save=SaveModel()

#Preprocessing

"""
data_raw,label_raw=[],[]
folders=['man/','woman/']
for i in range(2):
    for file in listdir(folders[i]):
        data_raw.append(imread(folders[i]+file));label_raw.append(i)

data,label=[],[]
while len(data_raw)>0:
    index=randint(0,len(data_raw)-1)
    data.append(data_raw[index]);label.append(label_raw[index])
    del data_raw[index],label_raw[index]
data,label=np.array(data),np.array(label)
label=to_categorical(label,num_classes=2)


data,label=np.load('data.npy'),np.load('label.npy')

"""
# In[6]:


#Model

model=Sequential([
    Conv2D(input_shape=(32,32,3),kernel_size=5,filters=12,padding='same',activation='relu'),
    MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    Dropout(0.1),
    Conv2D(kernel_size=5,filters=12,padding='same',activation='relu'),
    MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    Dropout(0.1),
    Conv2D(kernel_size=5,filters=12,padding='same',activation='relu'),
    MaxPooling2D(pool_size=(2,2),strides=(2,2)),
    Flatten(),
    Dense(activation='softmax',units=2)
])

model.compile(optimizer=Adadelta(lr=1e-3),metrics=['accuracy'],loss='categorical_crossentropy')


# In[ ]:


#Train

model.fit(data,label,epochs=2000,batch_size=470)
model.save('genderRecog.h5')
cost,accuracy=model.evaluate(data[:100],label[:100])
print('Cost: {}\nAccuracy: {}'.format(cost,accuracy))

