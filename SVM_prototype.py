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

from itertools import combinations
#================End Importing section=====================

# check if tkinter working
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')

# 101 is main dataset, 102 is for testing
parent_dir = "/mnt/c/Users/nhmin/Downloads/food-101/"
bin_n = 16 # Number of bin
# svm_params = dict(kernel_type= cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=2.67, gamma=5.383)

def get_image(path):
    # image size = 512x512
    img = Image.open(path + ".jpg")
    resized_image = img.resize((384,384), Image.ANTIALIAS)
    return np.array(resized_image)

# Currently testing to get features
# def load_train_data():
#     final_data = dict()
#     # Load in json file to create dictionary: key = class label; value = file path
#     fileName = parent_dir + "/meta/train.json"
#     with open(fileName, 'r') as file:
#         data = json.load(file)
#     # get label list
#     labels = list(data.keys())
#     file_image = get_image(parent_dir + "images/" + data.get(labels[0])[0])
#     orb = cv2.ORB()

#     kp = orb.detect(file_image, None)
#     kp, des = orb.compute(file_image, kp)
#     print(kp)
#     print("==================")
#     print(des)

def load_json(path):
    final_data = dict()
    # Load in json file to create dictionary: key = class label; value = file path
    with open(path, 'r') as file:
        data = json.load(file)
    return data


def load_data(data):
    final_data = dict()
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
    return final_data

def load_HOG_data(data):
    feature_lists = []
    for image in data:
            file_name = image.split('/')[1]
            file_image = get_image(parent_dir + "images/" + image)
            # given 32x32 cell
            image_feature, image_hog = hog(file_image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(8, 8), block_norm = 'L2-Hys', visualize=True, multichannel=True)
            feature_lists.append(image_feature)
    return np.array(feature_lists)

def label_processing(data, label_dict):
    final_label = []
    for i in label_dict.keys():
        label_value = label_dict.get(i)
        for image in data.get(i):
            final_label.append(label_value)
    return np.array(final_label)

def feature_format(data):
    feature_matrix = load_HOG_data(data)
    ss = StandardScaler()
    food_stand = ss.fit_transform(feature_matrix)
    pca = PCA(n_components = feature_matrix.shape[0])
    food_pca = ss.fit_transform(food_stand)
    return food_pca

def main():
    # Compute both train and test data
    # load json file
    train_json = load_json(parent_dir + "/meta/train.json")
    test_json = load_json(parent_dir + "/meta/test.json")

    # Getting all the label
    labels = list(train_json.keys())
    class_combinations = list(combinations(labels, 2))

    # for combination in class_combination:
    #     class1 = load_HOG_data(train_json.get(combination[0]))
    #     class2 = load_HOG_data(train_json.get(combination[1]))

    # testing code
    combine_data_train = train_json.get(class_combinations[0][0]) + train_json.get(class_combinations[0][1])
    combine_data_test = test_json.get(class_combinations[0][0]) + test_json.get(class_combinations[0][1])
    label_dict = {class_combinations[0][0] : 1, class_combinations[0][1] : -1}
    train_pca = feature_format(combine_data_train)
    test_pca = feature_format(combine_data_test)

    label_lists_train = label_processing(train_json, label_dict)
    label_lists_test = label_processing(test_json, label_dict)

    X_Train = pd.DataFrame(train_pca)
    Y_Train = pd.Series(label_lists_train)

    X_Test = pd.DataFrame(test_pca)
    Y_Test = pd.Series(label_lists_test)

    #SVM classification
    svm = SVC(kernel='linear', probability=True)
    svm.fit(X_Train, Y_Train)

    Y_pred = svm.predict(X_Test)

    # Calculate accuracy
    accuracy = accuracy_score(Y_Test, Y_pred)
    print("Model accuracy: ", accuracy)


if __name__ == '__main__':
    main()


    