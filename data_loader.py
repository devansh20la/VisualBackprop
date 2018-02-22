# from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
from torchvision import transforms, utils
from PIL import Image
import random

class imageandlabel(Dataset):

    def __init__ (self,root_dir,csv_file,trans=None):
        # self.csvfile = pd.read_csv(csv_file)
        self.files = [fn for fn in os.listdir(root_dir) if fn.endswith('.jpg')]

        self.trans = trans
        self.root_dir = root_dir

    def __len__(self):
        return len(self.files)
        # return len(self.csvfile)

    def __getitem__(self,idx):
        # imgname = os.path.join(self.root_dir,self.csvfile.iloc[idx,0])
        imgname = os.path.join(self.root_dir,self.files[idx])
        # image = Image.open(imgname+'.jpg')
        image = Image.open(imgname)
        label = 0
        # self.csvfile.ix[idx,1]
        prediction = 0
        # self.csvfile.ix[idx,2]

        if self.trans:
            image = self.trans(image)

        sample = {'image': image, 'label': label, 'prediction': prediction,'path': self.files[idx]}
        # self.csvfile.iloc[idx,0]}

        return sample

