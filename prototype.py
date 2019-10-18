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
classes = ["bibimbap","falafel"]

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


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

dir = "/Users/flo/Desktop/456/project/food-101/images_test"
classes = ["bibimbap","donuts"]

for cl in classes:
    path = os.path.join(dir, cl) # path = /Users/flo/Desktop/456/project/food-101/images/bibimbap
    #print(path)
    label = classes.index(cl)
    print(classes.index(cl))
    input('wait')
    for img in os.listdir(path): # img = 3845303.jpg
        if img.startswith('.'): # .DS_Store
            break
        try: 
            img_bgr = cv2.imread(os.path.join(path,img)) # imread takes in path to foodpic.jpg
            img_rgb = img_bgr[:, :, ::-1] # imread defaults to BGR, so convert into RGB
            plt.imshow(img_rgb)
            plt.show() 
            #training_data.append([img_rgb, label])
        except Exception as e:
            pass
        #print(img_rgb.shape) # (512, 384, 3) => 3845303.jpg is 384 x 512 dimensions
        #input('wait')
        


# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

for cl in classes:
    path = os.path.join(dir, cl) # path = /Users/flo/Desktop/456/project/food-101/images/bibimbap
    label = classes.index(cl)
    for img in os.listdir(path): # img = 3845303.jpg
        if img.startswith('.'): # .DS_Store
            break
        try: 
            img_bgr = cv2.imread(os.path.join(path,img)) # imread takes in path to foodpic.jpg
            img_rgb = img_bgr[:, :, ::-1] # imread defaults to BGR, so convert into RGB
            #plt.imshow(img_rgb)
            #plt.show() 
            print('appending',img_rgb)
            training_data.append([img_rgb, label])
        except Exception as e:
            pass
        break
    break
        #print(img_rgb.shape) # (512, 384, 3) => 3845303.jpg is 384 x 512 dimensions
        #input('wait')
    


# In[6]:


dir = "/Users/flo/Desktop/456/project/food-101/images_test/bibimbap/3924141.jpg"

img = cv2.imread(dir)

print(img.shape)
print(img)


# In[4]:


IMG_SIZE = 100

# new_img = cv2.resize(img_rgb, (IMG_SIZE,IMG_SIZE))
# plt.imshow(new_img)
# plt.show()  
# print(new_img.shape)


# In[9]:


import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

training_data = []

def create_training_data(dir, classes):
    for cl in classes:
        path = os.path.join(dir, cl) # path = /Users/flo/Desktop/456/project/food-101/images/bibimbap
        label = classes.index(cl)
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

dir = "/Users/flo/Desktop/456/project/food-101/images_test"
classes = ["bibimbap","donuts"]

td = create_training_data(dir, classes)
print(td)


# In[12]:


import random
random.shuffle(td)


# In[16]:


for s in td:
    print(s[1])


# In[27]:


X = []
y = []

for features, label in td:
    X.append(features)
    y.append(label)
    
#print('X',X)
#print('y',y)

X = np.array(X).reshape(-1, IMG_SIZE, IMG_SIZE, 3) # -1 = any number of features
print(X)

