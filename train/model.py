import os
import os.path as ops
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import models

class resnet18_encoder(nn.Module):
    def __init__(self):
        super(resnet18_encoder, self).__init__()
        resnet18 = models.resnet18(pretrained=True)
        resnet18_layers = list(resnet18.children())[:-1] 
        self.resnet18 = nn.Sequential(*resnet18_layers)
        self.fc1 = nn.Linear(512, 256)

    def forward(self, x):
        output = self.resnet18(x)
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        output = F.normalize(output, p=2, dim=1)
        return output

class densenet121_encoder(nn.Module):
    def __init__(self):
        super(densenet121_encoder, self).__init__()
        densenet = models.densenet121(pretrained=True)
        densenet_layers = list(densenet.children())[:-1] 
        self.densenet = nn.Sequential(*densenet_layers)
        self.fc1 = nn.Linear(1024, 256)

    def forward(self, x):
        output = self.densenet(x)
        output = F.relu(output, inplace=True)
        output = F.adaptive_avg_pool2d(output,(1,1))
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        output = F.normalize(output, p=2, dim=1)
        return output

class mobilenet_encoder(nn.Module):
    def __init__(self):
        super(mobilenet_encoder, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        mobilenet_layers = list(mobilenet.children())[:-1] 
        self.mobilenet = nn.Sequential(*mobilenet_layers)
        self.fc1 = nn.Linear(1280, 256)

    def forward(self, x):
        output = self.mobilenet(x)
        output = nn.functional.adaptive_avg_pool2d(output, 1).reshape(output.shape[0], -1)
        output = torch.flatten(output, 1)
        output = self.fc1(output)
        output = F.normalize(output, p=2, dim=1)
        return output



class swin_v2_t_encoder(nn.Module):
    def __init__(self):
        super(swin_v2_t_encoder, self).__init__()
        self.model = models.swin_v2_t(pretrained=True)
        self.model.head = nn.Linear(in_features=768, out_features=256, bias=True)

    def forward(self, x):
        output = self.model(x)
        print(self.model)
        print(output.shape)
        #output = nn.functional.adaptive_avg_pool2d(output, 1).reshape(output.shape[0], -1)
        output = torch.flatten(output, 1)
        #output = self.fc1(output)
        output = F.normalize(output, p=2, dim=1)
        return output



class efficientnet_b0_encoder(nn.Module):
    def __init__(self):
        super(efficientnet_b0_encoder, self).__init__()
        self.model = torchvision.models.efficientnet_b0(pretrained=True)
        self.model.classifier[1] = nn.Linear(in_features=1280, out_features=256, bias=True)

    def forward(self, x):
        output = self.model(x)
        #output = nn.functional.adaptive_avg_pool2d(output, 1).reshape(output.shape[0], -1)
        output = torch.flatten(output, 1)
        #output = self.fc1(output)
        output = F.normalize(output, p=2, dim=1)
        return output



class regnet_x_400mf_encoder(nn.Module):
    def __init__(self):
        super(regnet_x_400mf_encoder, self).__init__()
        self.model = torchvision.models.regnet_x_400mf()
        self.model.fc = nn.Linear(in_features=400, out_features=256, bias=True)
    def forward(self, x):
        output = self.model(x)
        #output = nn.functional.adaptive_avg_pool2d(output, 1).reshape(output.shape[0], -1)
        output = torch.flatten(output, 1)
        #output = self.fc1(output)
        output = F.normalize(output, p=2, dim=1)
        return output



#elif args.encoder_arc == "swin_v2_t":#2022
#model = torchvision.models.swin_v2_t()
#elif args.encoder_arc == "efficientnet_b0":#2019
#model = torchvision.models.efficientnet_b0()
#elif args.encoder_arc == "regnet_x_400mf":
#model = torchvision.models.regnet_x_400mf()
#elif args.encoder_arc == "regnet_y_400mf":#2020
#model = torchvision.models.regnet_y_400mf()