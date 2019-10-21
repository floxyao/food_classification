#!/usr/bin/env python
# coding: utf-8

# In[ ]:




pip install opencv-python


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

dir = "/Users/flo/Desktop/456/project/food-101/images"
classes = ["apple_pie",
           "baby_back_ribs",
           "baklava",
           "beef_carpaccio",
           "beef_tartare",
           "beet_salad",
           "beignets",
           "bibimbap",
           "bread_pudding",
           "breakfast_burrito",
           "bruschetta",
           "caesar_salad",
           "cannoli",
           "caprese_salad",
           "carrot_cake",
           "ceviche",
           "cheese_plate",
           "cheesecake",
           "chicken_curry",
           "chicken_quesadilla",
           "chicken_wings",
           "chocolate_cake",
           "chocolate_mousse",
           "churros",
           "clam_chowder",
           "club_sandwich",
           "crab_cakes",
           "creme_brulee",
           "croque_madame",
           "cup_cakes",
           "deviled_eggs",
           "donuts",
           "dumplings",
           "edamame",
           "eggs_benedict",
           "escargots",
           "falafel",
           "filet_mignon",
          "fish_and_chips"
          "foie_gras",
          "french_fries",
          "french_onion_soup",
          "french_toast",
          "fried_calamari",
          "fried_rice",
          "frozen_yogurt",
          "garlic_bread",
          "gnocchi",
          "greek_salad",
          "grilled_cheese_sandwich",
          "grilled_salmon",
          "guacamole",
          "gyoza",
          "hamburger",
          "hot_and_sour_soup",
          "hot_dog",
          "huevos_rancheros",
          "hummus",
          "ice_cream",
          "lasagna",
          "lobster_bisque",
          "lobster_roll_sandwich",
          "macaroni_and_cheese",
          "macarons",
          "miso_soup",
          "mussels",
          "nachos",
          "omelette",
          "onion_rings",
          "oysters",
          "pad_thai",
          "paella",
          "pancakes",
          "panna_cotta",
          "peking_duck",
          "pho",
          "pizza",
          "pork_chop",
          "poutine",
          "prime_rib"
          "pulled_pork_sandwich",
          "ramen",
          "ravioli",
          "red_velvet_cake",
          "risotto",
          "samosa",
          "sashimi",
          "scallops",
          "seaweed_salad",
          "shrimp_and_grits",
          "spaghetti_bolognese",
          "spaghetti_carbonara",
          "spring_rolls",
          "steak",
          "strawberry_shortcake",
          "sushi",
          "tacos",
          "takoyaki",
          "tiramisu",
          "tuna_tartare",
          "waffles"]

print (classes)

for cl in classes:
    path = os.path.join(dir, cl)
    for img in os.listdir(path):
        img_arr = cv2.imread(os.path.join(path,img)) 
        plt.imshow(img_arr)
        plt.show()
        break
    break


# In[ ]:


dir = "/Users/flo/Desktop/456/project/food-101/images_test/bibimbap/3924141.jpg"

img = cv2.imread(dir)

print(img.shape)
print(img)


# In[1]:


IMG_SIZE = 100

# new_img = cv2.resize(img_rgb, (IMG_SIZE,IMG_SIZE))
# plt.imshow(new_img)
# plt.show()  
# print(new_img.shape)


# In[19]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

training_data = []

def create_training_data(dir, classes):
    for cl in classes:
        path = os.path.join(dir, cl) # path = /Users/flo/Desktop/456/project/food-101/images/bibimbap
        label = classes.index(cl)
        print('label',label)
        for img in os.listdir(path): # img = 3845303.jpg
            #print('image',img)
            if img.startswith('.'): # .DS_Store
                continue
            #print(img,'good to go')
            try: 
                img_bgr = cv2.imread(os.path.join(path,img)) # imread takes in path to foodpic.jpg
                img_rgb = img_bgr[:, :, ::-1] # imread defaults to BGR, so convert into RGB
                new_img = cv2.resize(img_rgb, (IMG_SIZE,IMG_SIZE))
                
                #print('shape',new_img.shape)
                training_data.append([new_img, label]) # td is array of <100x100x3 img, label>
            except Exception as e:
                print("exception thrown: ",e)
            #input('wait')
    return training_data

dir = "/Users/flo/Desktop/456/project/food-101/images"
classes = ["apple_pie",
           "baby_back_ribs"
          ]

td = create_training_data(dir, classes)
print(td)


# In[13]:


import random
random.shuffle(td)


# In[14]:


for samples in td:
    print(sample[1])


# In[15]:


X = []
y = []

for features, label in td:
    X.append(features)
    y.append(label)
    
#print('X',X)
#print('y',y)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) # -1 = any number of features (catch-all, it'll recognize)
y = np.array(y).reshape(-1, 1)
# print(y.shape)
# print(X)
# print(X.shape) # (2,100,100,3) 2 is how many of those feature sets do we have


# In[16]:


import pickle

#save lists for future use 
pickle_out = open("X.pickle","wb")
pickle.dump(X, pickle_out)
pickle_out.close()

pickle_out = open("y.pickle","wb")
pickle.dump(y, pickle_out)
pickle_out.close()


# In[17]:


# pickle_in = open("X.pickle","rb")
# pickle_in = open("y.pickle","rb")
# print(X[1])
# y[1]


# In[18]:


# before we wanna feed data into neural network (X, y)
# we want to normalize the data
# eziest way is to scale data

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

X = pickle.load(open("X.pickle","rb"))
y = pickle.load(open("y.pickle","rb"))

# since we're taking in images, we know that the max of an image = 255 and min = 0
X = X/255.0 # scale; normally you use keras.normalize() something like that

# specify model
model = Sequential()

# layer 1: initial layer
            #   CNN, window, shape of data
model.add((Conv2D(64, (3,3), input_shape = X.shape[1:]))) # shape looks like (2, 100, 100, 3); dont need 2
model.add(Activation("relu")) # rectify linear; we can pass pooling or activation after CNN, doesn't matter
model.add(MaxPooling2D(pool_size=(2,2)))

# layer 2: pass again without input shape
model.add((Conv2D(64, (3,3))))
model.add(Activation("relu")) # rectify linear; we can pass pooling or activation after CNN, doesn't matter
model.add(MaxPooling2D(pool_size=(2,2)))

# layer 3: final layer is flaten and dense
model.add(Flatten())
model.add(Dense(64))

# layer 4: output layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss="binary_crossentropy",
             optimizer="adam",
             metrics=['accuracy'])

model.fit(X,y,batch_size=32,epochs=3, validation_split=0.1) # batch_size is how much data a time we wanna pass through a layer
                                      # epochs = how many times go through the network

