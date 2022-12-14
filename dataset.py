import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt

class QcellTrainDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.images.sort()
        self.label_list = list(range(-21,21,2))
        self.images_LT = []
        for i in range(len(self.images)):
            if i % 4 == 0:
                self.images_LT.append(self.images[i])

    def __len__(self):
        return len(self.images_LT)
    '''
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images_LT[index])
        image = plt.imread(image_path)
        label = self.label_list[(int(self.images[index][0:5])+1)//500]

        return image, label
    '''
    def __getitem__(self, index):
        image = np.zeros((111,111,4))
        image_path = os.path.join(self.image_dir, self.images_LT[index])
        image[:,:,0] = plt.imread(image_path)
        for i in range(1,4):
            image_path = os.path.join(self.image_dir, self.images[index*4+i])
            image[:,:,i] = plt.imread(image_path)
        label = self.label_list[(int(self.images[index][0:5])+1)//500]

        return image, label
    

class QcellValDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.images = os.listdir(image_dir)
        self.images.sort()
        self.label_list = list(range(-21,21,2))

    def __len__(self):
        return len(self.images)
    '''
    def __getitem__(self, index):
        image_path = os.path.join(self.image_dir, self.images[index])
        image = plt.imread(image_path)
        label = self.label_list[(int(self.images[index][0:5])+1)//500]

        return image, label
    '''
    def __getitem__(self, index):
        image = np.zeros((111,111,4))
        image_path = os.path.join(self.image_dir, self.images_LT[index])
        image[:,:,0] = plt.imread(image_path)
        for i in range(1,4):
            image_path = os.path.join(self.image_dir, self.images[index*4+i])
            image[:,:,i] = plt.imread(image_path)
        label = self.label_list[(int(self.images[index][0:5])+1)//500]

        return image, label
    
