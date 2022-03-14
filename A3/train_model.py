import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from os import name
import pickle
import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import numpy as np
import cv2
from torch.utils.data import random_split
import sys

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 11 * 11, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 36)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 11 * 11)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class TrainData(data.Dataset):

    def __init__(self):
        with open('comp551/images_l.pkl', 'rb') as f:
            self.data_train = pickle.load(f)

        with open('comp551/labels_l.pkl', 'rb') as f:
            self.data_label = pickle.load(f)
        
        self.HW = 56

        print(self.data_train.shape)
        print(self.data_label.shape)

    def __getitem__(self, index):

        # get one item
        data = self.data_train[index] # [56, 56]
        label = self.data_label[index] # [36, ]

        #cv2.imwrite('Original image.png', data)

        # print(data.shape)
        # print(label.shape)
        
        # Augmentation
        # flip (randomly)
        if np.random.rand(1)>0.5:
            data = cv2.flip(data, 0)
               
        if np.random.rand(1)>0.5:
            data = cv2.flip(data, 1)

        #cv2.imwrite('Fliped image.png', data)
        
        # rotate
        if np.random.rand(1)>0.5:
            M_2 = cv2.getRotationMatrix2D((28, 28), -90, 1)
            data = cv2.warpAffine(data, M_2, (56, 56))
        
        #cv2.imwrite('Rotated_-90.png', data)

        # denoise
        data = cv2.GaussianBlur(data,(5,5),1)

        #cv2.imwrite('Gaussian.png', data)
        
        # to Tenser
        data = data.reshape(-1, self.HW, self.HW) # [1, 56, 56]
        label = label.reshape(-1, 36) # [1, 36]
        data = torch.from_numpy(data.astype(np.float32) / 255.0)

        return data, label

    def __len__(self):
        return self.data_train.shape[0]

if __name__ == '__main__':       

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
    trainset = TrainData()
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)

    for epoch in range(10):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

        # zero the parameter gradients
            optimizer.zero_grad()

        # forward + backward + optimize
            outputs = net(inputs)
            # index = np.nonzero(outputs)
            # outputs = index[1][0]*26 + (index[1][1]-10)
            loss = criterion(outputs, labels.reshape(1, 36))
            # print(loss)
            loss.backward()
            optimizer.step()

        # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0


    print('Finished Training')


