import torch
from torch import nn


class Net(torch.nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.model = torch.nn.Sequential(
            #The size of the picture is 28x28
            torch.nn.Conv2d(in_channels = 1,out_channels = 4,kernel_size = 5,stride = 1,padding = 0),
            # 通道为4，24x24
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2),
            # 通道为4，12x12
            torch.nn.Conv2d(in_channels = 4,out_channels = 8,kernel_size = 5,stride = 1,padding = 0),
            # 通道为8，8x8
            # torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2),
            # 通道为8，4x4
            torch.nn.Flatten(),
            # torch.nn.Linear(in_features = 8*4*4,out_features = 8*4*4),
            # torch.nn.ReLU(),
            torch.nn.Linear(in_features = 8*4*4,out_features = 8),
            torch.nn.Linear(8,10),
            # torch.nn.Softmax(dim=1)
        )
        
    def forward(self,input):
        output = self.model(input)
        return output