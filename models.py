## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net_01(nn.Module):

    def __init__(self):
        super(Net_01, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # image size: (1, 224, 224)
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        
        # 1 input image channel (grayscale), 32 output channels/feature maps, 4x4 square convolution kernel (stride=1, zero-padding=0)
        self.conv1 = nn.Conv2d(1, 32, 4)
        self.conv1_bn = nn.BatchNorm2d(32)
        
        # output size: (32, 221, 221)
        
        # pool with kernel_size=2, stride=2
        self.pool1 = nn.MaxPool2d(2, 2)
        # output size will be (221-2)/2+1=110.5 => (32, 110, 110)  
        
        self.pool1_drop = nn.Dropout(p=0.1)
        
        # 32 input channels, 64 output channels, 3x3 square convolution kernel
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv2_bn = nn.BatchNorm2d(64)

        # output size: (64, 108, 108)
        self.pool2 = nn.MaxPool2d(2, 2)
        # output size: (64, 54, 54)
        self.pool2_drop = nn.Dropout(p=0.2)
        
        # 64 input channels, 128 output channels, 2x2 square convolution kernel
        self.conv3 = nn.Conv2d(64, 128, 2)
        self.conv3_bn = nn.BatchNorm2d(128)

        # output size: (128, 53, 53)
        self.pool3 = nn.MaxPool2d(2, 2)
        # output size: (128, 26, 26)
        self.pool3_drop = nn.Dropout(p=0.3)
        
        # 128 input channels, 256 output channels, 2x2 kernel
        self.conv4 = nn.Conv2d(128, 256, 2)
        self.conv4_bn = nn.BatchNorm2d(256)
        # output size: (256, 25, 25)
        self.pool4 = nn.MaxPool2d(2, 2)
        # output size: (256, 12, 12)
        self.pool4_drop = nn.Dropout(p=0.3)
        
        
        self.fc1 = nn.Linear(256*12*12, 6000)
        #self.fc1_bn = nn.BatchNorm1d(100)
        self.fc1_drop = nn.Dropout(p=0.3)
        
        self.fc2 = nn.Linear(6000, 1000)
        #self.fc2_bn = nn.BatchNorm1d(100)
        self.fc2_drop = nn.Dropout(p=0.3)
        
        self.fc3 = nn.Linear(1000, 1000)
        #self.fc3_bn = nn.BatchNorm1d(100)
        self.fc3_drop = nn.Dropout(p=0.3)
        
        self.fc4 = nn.Linear(1000, 136)
    
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # print(str(x.shape))
        
        x = self.conv1(x)
        x = self.pool1(F.relu(self.conv1_bn(x)))
        x = self.pool1_drop(x)
        
        x = self.conv2(x)
        x = self.pool2(F.relu(self.conv2_bn(x)))
        #print(str(x.shape))
        x = self.pool2_drop(x)
        
        x = self.conv3(x)
        x = self.pool3(F.relu(self.conv3_bn(x)))
        x = self.pool3_drop(x)
        
        x = self.conv4(x)
        x = self.pool4(F.relu(self.conv4_bn(x)))
        x = self.pool4_drop(x)
        
        # flatten
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))
        x = self.fc1_drop(x)
        
        x = F.relu(self.fc2(x))
        x = self.fc2_drop(x)
        
        x = F.relu(self.fc3(x))
        x = self.fc3_drop(x)
        
        x = self.fc4(x)
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
