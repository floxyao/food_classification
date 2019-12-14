# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 16:23:09 2019

@author: nhmin
"""

#================Importing section=========================
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import json
import os
import cv2
import pickle
import csv

from PIL import Image
from skimage.feature import hog
from skimage.color import rgb2grey
from skimage.filters import prewitt_h,prewitt_v, sobel

# Library for scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Library for LIBSVM
from libsvm.svmutil import *
from itertools import combinations
from skimage.data import camera

# Library fror tensorflow and keras
import tensorflow as tf
from tensorflow import keras
#from keras import Activation, BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D, Sequential, optimizers
import h5py
#================End Importing section=====================

#check if tkinter working
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

# 101 is main dataset, 102 is for testing with smaller dataset
parent_dir = "/mnt/c/Users/nhmin/Downloads/food-103"
bin_n = 16 # Number of bin
project_dir = os.getcwd()
model_dir = "/mnt/c/Users/nhmin/Downloads/"
class_label = {"pad_thai" : 0, "pho" : 1, "ramen" : 2, "spaghetti_bolognese" : 3, "spaghetti_carbonara" : 4}


# Return np img size 128x128
def get_image(path):
    # image resize to 384x384
    img = Image.open(path + ".jpg")
    resized_image = img.resize((227,227), Image.ANTIALIAS)
    return np.array(resized_image)


def load_json(path):
    final_data = dict()
    # Load in json file to create dictionary: key = class label; value = file path
    with open(path, 'r') as file:
        data = json.load(file)
    # Only get information from needed class
    for label in class_label:
        final_data.update({label : data.get(label)})
    return final_data

def HOG_image(image):
    np_image = get_image(parent_dir + "/images/" + image)
    # given 32x32 cell
    image_feature, image_hog = hog(np_image, orientations=8, pixels_per_cell=(8, 8),
        cells_per_block=(8, 8), block_norm = 'L2-Hys', visualize=True, multichannel=True)
    return np.array(image_feature)

def pca_transform(image):
    ss = StandardScaler()
    image_ss = ss.fit_transform(image)
    # Keep 90% of variance
    pca = PCA(0.9)
    image_pca = pca.fit_transform(image_ss)
    return image_pca

def get_key(val):
    for key, value in class_label.items():
        if(val == value): return key
    return None

# def load_train_data():
#     with h5py.File(model_dir + '227-imgsz-32-bsz-0.01-lr-30-ep.ckpt.02-1.64.hdf5', 'r') as f:
#         print(f.keys())
#         data = f['model_weights']
#         for name in data:
#             print(name)
#         print(data['max_pooling2d_18'])
#         # print(data.shape)
#         # print(data.dtype)
#         # print(f['model_weights'][0])
#         # print("===============")
#         # print(f['optimizer_weights'])

def training():
    # load json file
    train_json = load_json(parent_dir + "/meta/train.json")

    # Getting all the label and combination of all label
    # labels = list(train_json.keys())
    # class_combinations = list(combinations(labels, 2))

    # Create all train data after HOG
    train_image = []
    train_data = []
    train_label = []
    for label in train_json.keys():
        for img in train_json.get(label):
            # train_image.append(img.split('/')[1])
            # train_label.append(class_label.get(label))
            train_data.append(HOG_image(img))
    # Applying PCA 
    train_pca = pca_transform(train_data)
    print(len(train_data[1]))
    print(train_pca.shape)
    # train_model = svm_train(train_label, train_data, '-s 0 -t 0 -c 2.3 -b 1')
    # svm_save_model('all_food_classification.model', train_model)
    

def testing():
    # load json file
    test_json = load_json(parent_dir + "/meta/test.json")
    model = svm_load_model('all_food_classification.model')
    # Create all train data after HOG
    test_image = []
    test_data = []
    test_label = []
    for label in test_json.keys():
        for img in test_json.get(label):
            test_image.append(img.split('/')[1])
            test_label.append(class_label.get(label))
            test_data.append(HOG_image(img))
    
    p_label, p_acc, p_val = svm_predict(test_label, test_data, model, '-b 1')
    # acc, mse, scc = evaluations(test_label, p_label)
    #print("Test acc: ", acc)
    for i in range(0, len(test_label)):
        if(test_label[i] != p_label[i]):
            print(test_image[i])
            print("True label: ", get_key(test_label[i]), " Predict label: ", get_key(p_label[i]))

def show_pca(image_path):
    image_grey = Image.open(image_path + ".jpg").convert('L')
    image_grey.show()
    image_np = np.array(image_grey.resize((128,128), Image.ANTIALIAS))
    print(image_np.shape)
    image_pca = pca_transform(image_np)
    image_rgb = Image.fromarray(image_pca, 'L')
    image_rgb.show()




def single_test(image_path):
    #image_np = get_image(image_path)
    image_hog = HOG_image(image_path)
    print(image_hog)
    model = svm_load_model('all_food_classification.model')
    p_label, p_acc, p_val = svm_predict([], [image_hog], model, '-b 1')
    print(p_label)
    print(get_key(p_label[0]))
    print(p_val)    
# DON'T NEED FOR NOW
# def mean_pixel_image(image):
#     np_image = get_image(parent_dir + "images/" + image)
#     img_shape = np_image.shape
#     temp_img = np.zeros((img_shape[0], img_shape[1]))
#     for i in range(0, img_shape[0]):
#         for j in range(0, img_shape[1]):
#             temp_img[i][j] = (int(np_image[i,j,0]) + int(np_image[i,j,1]) + int(np_image[i,j,2]))/3
#     return np.array(np.reshape(temp_img, (img_shape[0],img_shape[1]))).ravel()

def main():
    #load_train_data()
    training()
    print("model training is finished")
    #testing()
    #print("Single image test.")
    #single_test("pho/34077")
    #show_pca(parent_dir + "images/pho/34077")


if __name__ == '__main__':
    main()


    