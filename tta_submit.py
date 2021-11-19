import torch
import os
import random
import numpy as np
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from Cyto_dataset import Cyto_train_dataset, Cyto_test_dataset
from torchvision import models
import torch.nn as nn
from utils import *
import torchvision
import torchvision.transforms.functional as TF
import scipy as sp
import time
from albumentations.core.transforms_interface import DualTransform, ImageOnlyTransform

import torchvision
import time

from classifier_utils_sw import submit_probs

seed=20
seed_everything(seed)


batch_size = 64
image_size = 320

class Test_Rotation(DualTransform):
    """for tta"""
    def __init__(self, angle, always_apply=False, p=0.5):
        self.angle = angle
        self.always_apply = always_apply
        self.p = p

        super(Test_Rotation, self).__init__(always_apply, p)
        self.mask_value = None

    def apply(self, image, **params):
        # print(image.shape)
        return sp.ndimage.rotate(image, self.angle, axes=(0,1), order=1,
                reshape=False, mode='reflect')

    def get_transform_init_args_names(self):
        return ()

class Test_Shift(DualTransform):
    """for tta"""
    def __init__(self, shift, always_apply=False, p=0.5):
        self.shift = shift + [0]
         # 0(h) or 1(w)
        self.always_apply = always_apply
        self.p = p

        super(Test_Shift, self).__init__(always_apply, p)
        self.mask_value = None

    def apply(self, image, **params):
        # print(image.shape)
        return sp.ndimage.shift(image, self.shift, order=1, mode='reflect')

    def get_transform_init_args_names(self):
        return ()


scale = A.Resize(image_size, image_size)
#rcrop = A.RandomCrop(width=256, height=256)
shift_w = Test_Shift([0,10], p=1)
shift_h = Test_Shift([10,0], p=1)
rotate1 = Test_Rotation(15, p=1)
rotate2 = Test_Rotation(-15, p=1)
H_flip = A.HorizontalFlip(p=1)
V_flip = A.VerticalFlip(p=1)

normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
to_tensor = ToTensorV2()


test_list = sorted(os.listdir('/home/ec2-user/dataset/b-trac-cyto/final/cyto_test_set'), key = lambda x : int(x.split('_')[-1].split('.')[0]))
for idx in range(len(test_list)):
    test_list[idx] = os.path.join('/home/ec2-user/dataset/b-trac-cyto/final/cyto_test_set', test_list[idx])
# print(test_list)

file_name = 'image_size_320_cos_full/14'

model = models.resnet50(pretrained=False)
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load( '/home/ec2-user/workspace/check_point/' + file_name + '.pth'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

make_csv = True
save_dir = './result/'
createFolder(save_dir)
print('batch_size', batch_size)
# transform_list = [None, V_flip, H_flip, rotate1, rotate2, shift_w, shift_h] # 0, 1, 2, 3, 4
transform_list = [None, V_flip, H_flip]
t_name_list = ['o', 'V', 'H']
t_name_len = len(t_name_list)
cur_t_name = 'm1'

def make_csv(pred, file_name, prefix='' ):
    prefix = prefix + '_tta_'
    results_df = pd.DataFrame(pred).transpose()
    res_dir = save_dir + prefix + file_name.split('/')[0]
    results_df.to_csv(res_dir + '_test.csv', header=False, index=False)
    print('make!!')


probs = 0 # 누적 probs
print('TTA submit')
for idx, new_transform in enumerate(transform_list):
    if new_transform is None:
        new_transform = [scale]
    elif new_transform == 's2':
        new_transform = [scale2]
    else:
        new_transform = [scale, new_transform]
    cur_t_name += t_name_list[idx]
    test_transform = A.Compose( new_transform +  [normalize, to_tensor] )
    print('transform', new_transform)
    test_set = Cyto_test_dataset(test_list, test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
    print(f'num_test : {len(test_loader)}')

    dic = submit_probs(model, test_loader, device)
    probs += dic['probs']
    print(len(probs))
    pred = np.where(probs/(idx+1) > 0.5, 1, 0)
    print(pred)
    make_csv(pred, file_name, cur_t_name)

print('TTA submit2')

file_name2 = 'image_size_320_cos/14'

model = models.resnet50(pretrained=False)
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load( '/home/ec2-user/workspace/check_point/' + file_name2 + '.pth'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

scale = A.Resize(344, 344)
transform_list = [None, H_flip]
t_name_list = ['o', 'H']
cur_t_name += '_m2'

for idx, new_transform in enumerate(transform_list):
    if new_transform is None:
        new_transform = [scale]
    elif new_transform == 's2':
        new_transform = [scale2]
    else:
        new_transform = [scale, new_transform]
        
    cur_t_name += t_name_list[idx]
    test_transform = A.Compose( new_transform +  [normalize, to_tensor] )
    print('transform', new_transform)
    test_set = Cyto_test_dataset(test_list, test_transform)
    test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)
    print(f'num_test : {len(test_loader)}')

    dic = submit_probs(model, test_loader, device)
    probs += dic['probs']
    pred = np.where(probs/(idx+1+t_name_len) > 0.5, 1, 0)
    make_csv(pred, file_name2, cur_t_name)

print("done")