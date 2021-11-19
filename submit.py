import torch
import os
import random
import numpy as np
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2, ToTensor
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

import torchvision
import time

from classifier_utils_sw import submit_probs

seed=20
seed_everything(seed)


batch_size = 1
image_size = 320


scale = A.Resize(image_size, image_size)


normalize = A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
to_tensor = ToTensor()


test_list = os.listdir('/home/ec2-user/dataset/b-trac-cyto/final/cyto_test_set')


#file_name = 'res152-224-ssr-sgd_epoch23'

#model = models.resnet152(pretrained=False)
#model.fc = nn.Linear(model.fc.in_features,2)


model = models.resnext50_32x4d(pretrained=True)
num_ftrs = model.fc.in_features
model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
model.fc = nn.Linear(num_ftrs, 2)

model.load_state_dict(torch.load( '/workspace/check_point/image_size_320_cos/14.pth'))
use_cuda = torch.cuda.is_available()

device = torch.device("cuda" if use_cuda else "cpu")
model.to(device)

make_csv = True
save_dir = './result/'
createFolder(save_dir)
print('batch_size', batch_size)

test_transform = A.Compose(
        [
            scale,
            #scale_te,
            normalize,
            to_tensor
        ])

def make_csv(pred, file_name, prefix='' ):
    prefix = prefix + '_tta_'
    results_df = pd.DataFrame(pred).transpose()
    results_df.to_csv(save_dir + prefix + file_name + '.csv', header=False, index=False)
    print('make!!')
    
test_set = Cyto_test_dataset(test_list, test_transform)
test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=4)

dic = submit_probs(model, test_loader, device)
probs = dic['probs']
print(probs.shape)
pred = np.argmax(probs, 1)
make_csv(pred, file_name, cur_t_name)




print("done")