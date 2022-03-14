from os import name
import pickle
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import cv2
import sys

def dimension_change(label):
    digit = np.argmax(label[:10])
    alpha = np.argmax(label[-26:])
    new_index = digit*10+alpha
    new_label = np.zeros(260)
    new_label[new_index]=1
    return new_label

def label_change(x):
    dim = len(x)
    array = np.zeros([dim, 260])
    for i in range(len(x)):
        new_label = dimension_change(x[i])
#         np.concatenate((array, new_label), axis=0)
        array[i] = new_label
    return array

class TrainData(data.Dataset):

    def __init__(self):
        with open('comp551/images_test.pkl', 'rb') as f:
            self.data_train = pickle.load(f)

        with open('comp551/labels_l.pkl', 'rb') as f:
            self.data_label = pickle.load(f)

        print(self.data_label[:10])
        sys.exit()
        
        self.HW = 56

        print(self.data_train.shape)
        print(self.data_label.shape)

    def __getitem__(self, index):

        # get one item
        data = self.data_train[index] # [56, 56]
        label = self.data_label[index] # [36, ]
        print(label)

        # cv2.imshow('image.png', data)
        # cv2.waitKey(0)
        # sys.exit()

        #cv2.imwrite('Original image.png', data)

        # print(data.shape)
        # print(label.shape)
        
        # # Augmentation
        # # flip (randomly)
        # if np.random.rand(1)>0.5:
        #     data = cv2.flip(data, 0)
               
        # if np.random.rand(1)>0.5:
        #     data = cv2.flip(data, 1)

        # #cv2.imwrite('Fliped image.png', data)
        
        # # rotate
        # if np.random.rand(1)>0.5:
        #     M_2 = cv2.getRotationMatrix2D((28, 28), -90, 1)
        #     data = cv2.warpAffine(data, M_2, (56, 56))
        
        # #cv2.imwrite('Rotated_-90.png', data)

        # denoise
        data = cv2.GaussianBlur(data,(5,5),1)

        #cv2.imwrite('Gaussian.png', data)
        
        # to Tenser
        data = data.reshape(-1, self.HW, self.HW) # [1, 56, 56]
        label = label.reshape(-1, 36) # [1, 36]
        print(label)
        data = torch.from_numpy(data.astype(np.float32) / 255.0)
        label = label_change(label)
        print(label)
        sys.exit()

        return data, label

    def __len__(self):
        return self.data_train.shape[0]


class TestData(data.Dataset):

    def __init__(self):
        with open('comp551/images_test.pkl', 'rb') as f:
            self.data_test = pickle.load(f)

        print(self.data_test.shape)

    def __getitem__(self, index):

        # get one item
        data = self.data_test[index] # [1, 56, 56]
        

       # denoise
        data = cv2.GaussianBlur(data,(5,5),1)

        #cv2.imwrite('Gaussian.png', data)
        
        # to Tenser
        data = data.reshape(-1, self.HW, self.HW) # [1, 56, 56]
        data = torch.from_numpy(data.astype(np.float32) / 255.0)

        return data

    def __len__(self):
        return self.data_test.shape[0]
