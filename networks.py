import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import numpy as np

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 5)
        self.conv2 = nn.Conv2d(15, 20, 5)
        self.conv3 = nn.Conv2d(20, 25, 5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(600,200)
        self.fc2 = nn.Linear(200,2)

    def forward(self,x):
        x = x.unsqueeze(1)
        x = self.conv1(x) #conv2d1

        x = F.relu(x) #activation1
        x = self.pool(x) #maxpooling2d1

        x = self.conv2(x) #conv2d2
        x = F.relu(x) #activation2
        x = self.pool(x) #maxpooling2d2

        x = self.conv3(x) #conv2d3
        x = F.relu(x) #activation3
        x = self.pool(x) #maxpooling2d3

        x = x.view(x.size(0), -1) #flatten1
        x = self.fc1(x) #dense1
        x = F.relu(x) #activation4

        x = self.fc2(x)
        return x

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.conv1 = nn.Conv2d(1, 15, 7)
        self.conv2 = nn.Conv2d(15, 30, 5)
        self.conv3 = nn.Conv2d(30, 25, 3)
        self.conv4 = nn.Conv2d(25, 20, 7)
        self.conv5 = nn.Conv2d(20, 15, 5)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(225,150)
        self.fc2 = nn.Linear(150,116)

    def forward(self, x):
        # each layer should be followed by ReLU and optionally a maxpool layer
        x = x.unsqueeze(1)
        x = self.conv1(x) # conv2d1

        x = F.relu(x)  # activation1
        #x = self.pool(x)  # maxpooling2d1

        x = self.conv2(x)  # conv2d2
        x = F.relu(x)  # activation2
        x = self.pool(x)  # maxpooling2d2

        x = self.conv3(x)  # conv2d3
        x = F.relu(x)  # activation3
        x = self.pool(x)  # maxpooling2d3

        x = self.conv4(x)  # conv2d4
        x = F.relu(x)  # activation4
        x = self.pool(x)  # maxpooling2d4

        x = self.conv5(x)  # conv2d5
        x = F.relu(x)  # activation5
        x = self.pool(x)  # maxpooling2d5

        x = x.view(x.size(0), -1)  # flatten1
        x = self.fc1(x)  # dense1
        x = F.relu(x)  # activation4

        x = self.fc2(x)
        return x

    def show_filters(self):
        plt.imshow(self.conv1, cmap='gray')
        plt.show()

        # print(self.conv1.shape)
        # # 2 channels i think
        # for i in range(2):
        #     plt.imshow(self.conv1[:,i], cmap='gray')



class Network(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model_name = 'resnet18'
        self.model = models.resnet18()
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.model(x)
        return x


def main():
    network = Net2()
    network.show_filters()

if __name__ == "__main__":
    main()