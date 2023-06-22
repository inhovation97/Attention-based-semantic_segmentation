import glob
import cv2
import random
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
# from torch.utils.tensorboard import SummaryWriter
import torchvision
import torch.nn.functional as F
from torchvision import transforms, datasets
from ..cbam import *


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels,mode_):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.mode = mode_
        
        if self.mode == 'None':
            pass
        elif self.mode == 'sigmoid':
            self.f = nn.Sigmoid()
        elif self.mode == 'relu':
            self.f = nn.ReLU(inplace=True)
        elif self.mode == 'leakyrelu':
            self.f = nn.LeakyReLU(negative_slope=0.01, inplace=True)
        elif self.mode == 'tanh':
            self.f = nn.Tanh()
        elif self.mode == 'selu':
            self.f = nn.SELU(inplace=True) 
        elif self.mode == 'gelu':
            self.f = nn.GELU(approximate='none')    

    def forward(self, x):
        if self.mode == 'None':
            return self.conv(x)
        else:
            return self.f(self.conv(x))
    
    

    
    
class Attention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Attention, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels=1, kernel_size=1)
        self.tanh = nn.Tanh() 
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        xt = self.conv(x)
        xt = self.tanh(xt)
        xs = self.conv(x)
        xs = self.sigmoid(xs)
        
        return x * xt * xs



def deeplabv3_resnet101(outchannels=1, mode_='None', att=False, act_types =['sigmoid'], use_selu=False,alpha=1.0):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(42)
    if device == 'cuda':
          torch.cuda.manual_seed_all(42)

    model = torchvision.models.segmentation.deeplabv3_resnet101(weights='COCO_WITH_VOC_LABELS_V1')
    if att == 'cbam':
        model.classifier[4]=nn.Sequential(
            CBAM(256,act_types=act_types,alpha=alpha),
            OutConv(256,outchannels,mode_=mode_) 
            )
        
    else:
        if att:        
            model.classifier[4]=nn.Sequential(
                Attention(256,1),
                OutConv(256,outchannels,mode_=mode_) 
                )
        
        else :
            model.classifier[4]= OutConv(256,outchannels,mode_=mode_)
        

    
    if use_selu:
        model.classifier[3]=nn.SELU()
             
    return model