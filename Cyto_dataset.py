#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# In[ ]:


import os
import torch 
# import pandas as pd
import random
# import cv2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
import json
from utils import *

#/mnt/hackerton/dataset/Dataset/Eye/Train/Epiphora/0
#/mnt/hackerton/dataset/Dataset/Eye/label_data/Train/Epiphora/0

class Cyto_train_dataset(Dataset):
    def __init__(self, data_list, transform=[]):
        
        self.data_list = data_list
#         self.label_dir = os.path.join(data_dir, 'label_data/Train')
        
        self.transform = transform
        
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        '''
        '''
        image_path = self.data_list[idx]
        
        if not os.path.exists(image_path):
            print('dose not exist '+image_path)
            
        label_str = image_path.split('/')[6].split('_')[1]
#         print('label:',label_str)
        label = label_trans(label_str)
        
        image = Image.open(image_path)
        
#         if self.ben_prepro:
#             image = cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)
        
        if self.transform:
            augmented = self.transform(image=np.array(image))
            image = augmented['image']


#         if self.transform:
#             image = self.transform(image)

        sample = {'image':image, 'label':label}

        return sample

class Cyto_test_dataset(Dataset):
    def __init__(self, data_list, transform=[]):
        
        self.data_list = data_list
#         self.label_dir = os.path.join(data_dir, 'label_data/Train')
        
        self.transform = transform
        

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):

        image_path = self.data_list[idx]
        
        if not os.path.exists(image_path):
            print('dose not exist '+image_path)
        image = Image.open(image_path)
        
        
#         if self.ben_prepro:
#             image = cv2.addWeighted (image,4, cv2.GaussianBlur( image , (0,0) , 10) ,-4 ,128)
        
        if self.transform:
            augmented = self.transform(image=np.array(image))
            image = augmented['image']

#         if self.transform:
#             image = self.transform(image)


        sample = {'image':image}

        return sample

