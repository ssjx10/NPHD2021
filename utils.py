#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import torch
import numpy as np
import random

def label_trans(label_str):
    if label_str == 'positive':
        return 1
    elif label_str == 'negative':
        return 0
    else : print('label error')
        

    
    
    
    
def train_val_split(datadir, train_ratio=0.9):
    '''
    datadir = '/home/ec2-user/dataset/b-trac-cyto/final'
    '''
    
    neg_dir = os.path.join(datadir,'cyto_negative')
    pos_dir = os.path.join(datadir,'cyto_positive')
    
    neg_listdir = os.listdir(neg_dir)
    pos_listdir = os.listdir(pos_dir)
    
    
    for idx in range(len(neg_listdir)):
        neg_listdir[idx] = os.path.join(neg_dir, neg_listdir[idx])
    for idx in range(len(pos_listdir)):
        pos_listdir[idx] = os.path.join(pos_dir, pos_listdir[idx])
    
    random.shuffle(neg_listdir)
    random.shuffle(pos_listdir)
    
    print('neg_listdir : ', len(neg_listdir), 'pos_listdir : ', len(pos_listdir))
    
    len_train_neg = int(len(neg_listdir)*train_ratio)
    len_train_pos = int(len(pos_listdir)*train_ratio)
    
    train = neg_listdir[0:len_train_neg] + pos_listdir[0:len_train_pos]
    val = neg_listdir[len_train_neg:] + pos_listdir[len_train_pos:]
    
    random.shuffle(train)
    random.shuffle(val)
    
    print('len train : ', len(train), 'len val : ', len(val))
    
    return train, val
    
def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    #print(random.random())
    if torch.cuda.is_available():
        print(f'seed : {seed_value}')
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# In[ ]:




