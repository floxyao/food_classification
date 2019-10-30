# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:23:09 2019

@author: nhmin
"""

#================Importing section=========================
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import cv2

from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2grey

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc
#================End Importing section=====================

# check if tkinter working
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')


parent_dir = "/mnt/c/Users/nhmin/Downloads/food-101/"
bin_n = 16 # Number of bin

def get_image(path):
    # image size = 512x512
    img = Image.open(path + ".jpg")
    wid, hei = img.size
    return np.array(img)

def load_train_data():
    final_data = dict()
    # Load in json file to create dictionary: key = class label; value = file path
    fileName = parent_dir + "/meta/train.json"
    with open(fileName, 'r') as file:
        data = json.load(file)
    # get label list
    labels = list(data.keys()) 
    for label in labels:
        final_data.setdefault(label, [])
        for image in data.get(label):
            file_name = image.split('/')[1]
            file_image = get_image(parent_dir + "images/" + image)
            # given 32x32 cell
            image_feature, image_hog = hog(file_image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(8, 8), block_norm = 'L2-Hys', visualize=True, multichannel=True)
            final_data.get(label).append([file_name, image_feature])
            print(image)

    print(final_data)
    # plt.imshow(image(parent_dir + "images/" + data.get(label[0])[0]))
    # plt.show()
        
    #plt.imshow(image(parent_dir + "images/" + data.get(label[2])[4]))
    #plt.show()
    #print(data)
load_train_data()

    