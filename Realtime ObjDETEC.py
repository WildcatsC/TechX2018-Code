#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 20:47:52 2018

@author: stevenchen
"""
from keras.preprocessing.image import load_img, img_to_array

import tensorflow as tf
from keras.applications import imagenet_utils
from keras.applications import ResNet50, InceptionResNetV2, MobileNetV2, Xception
import cv2, threading
import numpy as np
from keras.models import load_model


# Initialize global variables to be used by the classification thread
# and load up the network and save it as a tensorflow graph
frame_to_predict = None
classification = True
label = ''
score = .0

print('Loading network...')
# model = VGG16(weights='imagenet')
model = load_model("shoe_model.h5")
graph = tf.get_default_graph()
print('Network loaded successfully!')

"""class MyThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        
    def run(self):
        global label
        global frame_to_predict
        global classification
        global model
        global graph
        global score
        
        with graph.as_default():
        
            while classification is True: 
                if frame_to_predict is not None:
                    
                    frame_to_predict = cv2.cvtColor(frame_to_predict, cv2.COLOR_BGR2GRAY).astype(np.float32)
                    frame_to_predict = frame_to_predict.reshape((28,28,1))
                    #frame_to_predict = cv2.cvtColor(frame_to_predict, cv2.COLOR_BGR2GRAY)
                    print(frame_to_predict.shape)
                    #frame_to_predict = frame_to_predict.reshape((1,)+frame_to_predict.shape)
                    frame_to_predict = imagenet_utils.preprocess_input(frame_to_predict)
                    predictions = model.predict(frame_to_predict)
                    (imageID, label, score) = imagenet_utils.decode_predictions(predictions)[0][0]


# Start a keras thread which will classify the frame returned by openCV
keras_thread = MyThread()
keras_thread.start()"""

# Initialize OpenCV video captue
video_capture = cv2.VideoCapture(0) 
video_capture.set(4, 800) 
video_capture.set(5, 600) 


img_rows, img_cols = 28, 28



while (True):
    
    ret, original_frame = video_capture.read()
    
    
    frame_to_predict = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)
    #print(frame_to_predict.shape)
    frame_to_predict = cv2.resize(frame_to_predict,(28,28)).reshape((1,28,28,1))
    #frame_to_predict = frame_to_predict.resize(frame_to_predict,(28,28))
    #frame_to_predict = frame_to_predict.reshape((28,28,1))
    #frame_to_predict = frame_to_predict.resize((28,28,1))
    #print(frame_to_predict.shape)
    
    label=model.predict(frame_to_predict)
    #print(label.shape)
    if label[0,9]==1 or label[0,7]==1 or label[0,5]==1 or label[0,0]==1:
        Label="sneaker"
    else:
        Label="not sneaker"
    # Add text label and network score to the video captue
    cv2.putText(original_frame, " " , 
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)
    if Label!= "sneaker":
        cv2.putText(original_frame, "FAKE!!!", (150,400), cv2.FONT_HERSHEY_SIMPLEX, 10.0, (0, 0, 255), 40)
    else: 
        cv2.putText(original_frame, "REAL", (420,400), cv2.FONT_HERSHEY_SIMPLEX, 5.0, (255,0,0),18)
    # Display the video
    
    cv2.imshow("Classification", original_frame)

    # Hit q or esc key to exit
    if (cv2.waitKey(1) & 0xFF == ord('q')):
        break;
        
classification = False
video_capture.release()
cv2.destroyAllWindows()



