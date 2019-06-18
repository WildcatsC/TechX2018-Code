#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 18:56:29 2018

@author: stevenchen
"""


pos_im_path = '../data/images/pos_person'
neg_im_path = '../data/images/neg_person'
min_wdw_sz = [68, 124]
step_size = [10, 10]
orientations = 9
pixels_per_cell = [6, 6]
cells_per_block = [2, 2]
visualize = False
normalize = True
pos_feat_ph = '../data/features/pos'
neg_feat_ph = '../data/features/neg'
model_path = '../data/models/'
threshold = .3