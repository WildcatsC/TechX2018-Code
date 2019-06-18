#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  3 22:21:12 2018

@author: stevenchen
"""

import numpy as np
import cv2
video_capture=cv2.VideoCapture(0)

while True:
    
    ret, original_frame = video_capture.read()
    new_frame=cv2.cvtColor(original_frame,cv2.COLOR_BGR2GRAY)
    new_frame=new_frame.reshape((720,1280,1))
    print(new_frame.shape)