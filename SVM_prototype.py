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
import pickle

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

# 101 is main dataset, 102 is for testing with smaller dataset
parent_dir = "/mnt/c/Users/nhmin/Downloads/food-101/"
bin_n = 16 # Number of bin
project_dir = os.getcwd()
model_dir = "/mnt/c/Users/nhmin/Downloads/"
class_label = ["pad_thai", "pho", "ramen", "spaghetti_bolognese", "spaghetti_carbonara"]
# svm_params = dict(kernel_type= cv2.SVM_LINEAR, svm_type=cv2.SVM_C_SVC, C=2.67, gamma=5.383)

def get_image(path):
    # image size = 512x512
    img = Image.open(path + ".jpg")
    resized_image = img.resize((384,384), Image.ANTIALIAS)
    return np.array(resized_image)


def load_json(path):
    final_data = dict()
    # Load in json file to create dictionary: key = class label; value = file path
    with open(path, 'r') as file:
        data = json.load(file)
    for label in class_label:
        final_data.update({label : data.get(label)})
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

def label_processing(data, label_dictionary):
    final_label = []
    for i in label_dictionary.keys():
        label_value = label_dictionary.get(i)
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

def pre_process_data(data_json):
    pca_data = dict()
    labels = list(data_json.keys())
    for label in labels:
        pca_data.update({label : feature_format(data_json.get(label))})
    return pca_data

def train_dataset(train_json):
    # Getting all the label and combination of all label
    labels = list(train_json.keys())
    class_combinations = list(combinations(labels, 2))

    #Get all pca dictinary data for both train and test
    train_pca_dict = pre_process_data(train_json)
    for combination in class_combinations:
        #SVM classification
        svm = SVC(gamma='auto', kernel='linear', probability=True)
        combine_train_data = np.vstack((train_pca_dict.get(combination[0]), train_pca_dict.get(combination[1])))
        label_dict = {combination[0] : 1, combination[1] : -1}
        label_lists_train = label_processing(train_json, label_dict)

        x_train = pd.DataFrame(combine_train_data)
        y_train = pd.Series(label_lists_train)

        svm.fit(x_train, y_train)
        filename = combination[0] + "_" + combination[1] + "_model.sav"
        model_filename = model_dir + "/models_save/" + filename
        pickle.dump(svm, open(model_filename, 'wb'))
def test_dataset(test_json):
    # Getting all the label and combination of all label
    labels = list(test_json.keys())
    class_combinations = list(combinations(labels, 2))
    #Get all pca dictinary data for both train and test
    test_pca_dict = pre_process_data(test_json)

    for combination in class_combinations:
        # load in SVM model
        filename = combination[0] + "_" + combination[1] + "_model.sav"
        loaded_model = pickle.load(open(model_dir + filename, 'rb'))
        combine_test_data = np.vstack((test_pca_dict.get(combination[0]), test_pca_dict.get(combination[1])))
        label_dict = {combination[0] : 1, combination[1] : -1}
        label_lists_test = label_processing(test_json, label_dict)

        x_test = pd.DataFrame(combine_test_data)
        y_test = pd.Series(label_lists_test)

        accuracy = loaded_model.score(x_test, y_test)
        y_pred = loaded_model.predict(x_text)
        print(combination)
        print("Test accuracy: " + accuracy)
        print("Test set prediction: " + y_pred)


def main():
    # Compute both train and test data
    # load json file
    train_json = load_json(parent_dir + "/meta/train.json")
    test_json = load_json(parent_dir + "/meta/test.json")

    train_dataset(test_json)
    test_dataset(test_json)

    # Getting all the label and combination of all label
    # labels = list(train_json.keys())
    # class_combinations = list(combinations(labels, 2))
    # #Get all pca dictinary data for both train and test
    # train_pca_dict = pre_process_data(train_json)
    # test_pca_dict = pre_process_data(test_json)

    # for combination in class_combinations:
    #     print(combination)
    #     #SVM classification
    #     svm = SVC(C=2.67,gamma='auto', kernel='linear', probability=True)
    #     combine_train_data = np.vstack((train_pca_dict.get(combination[0]), train_pca_dict.get(combination[1])))
    #     print(combine_train_data)
    #     print(combine_train_data.shape)
    #     combine_test_data = np.vstack((test_pca_dict.get(combination[0]), test_pca_dict.get(combination[1])))
    #     label_dict = {combination[0] : 1, combination[1] : -1}
    #     label_lists_train = label_processing(train_json, label_dict)
    #     label_lists_test = label_processing(test_json, label_dict)

    #     x_train = pd.DataFrame(combine_train_data)
    #     y_train = pd.Series(label_lists_train)

    #     x_test = pd.DataFrame(combine_test_data)
    #     y_test = pd.Series(label_lists_test)

    #     svm.fit(x_train, y_train)
    #     filename = combination[0] + "_" + combination[1] + "_model.sav"
    #     model_filename = model_dir + "/models_save/" + filename
    #     pickle.dump(svm, open(model_filename, 'wb'))

    #     y_pred = svm.predict(x_test)
    #     #calculate accuracy
    #     accuracy = accuracy_score(y_test, y_pred)
    #     print(combination)
    #     print("Model accuracy: ", accuracy)



if __name__ == '__main__':
    main()


    