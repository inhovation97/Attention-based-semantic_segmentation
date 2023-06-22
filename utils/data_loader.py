import os
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
import torchvision
from torchvision import transforms, datasets
from Codes.dataset import Dataset

def data_loader_resized(ROOT_PATH,BATCH_SIZE=8):
    # ROOT_PATH = '~YaML/data/Ottawa-Dataset/'
    
    DATASET={}
    LOADER={}
    
    _size = 224, 224
    
    # set your transforms
    train_transforms = transforms.Compose([
                               transforms.Resize(_size, interpolation=0),
                               transforms.RandomRotation(180),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(_size, padding = 10), # needed after rotation (with original size)
                           ])
    
    test_transforms = transforms.Compose([
                            
                               transforms.Resize(_size, interpolation=0),
                           ])
    
    trainset = Dataset(root_path = ROOT_PATH, mode = 'train', transforms = train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
    
    validset = Dataset(root_path = ROOT_PATH, mode = 'valid', transforms = None)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
    
    testset = Dataset(root_path = ROOT_PATH, mode = 'test', transforms = None)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    
    DATASET={'train': trainset,'valid': validset, 'test' : testset}
    LOADER={'train': train_loader,'valid' : valid_loader, 'test' : test_loader}
    
    return DATASET, LOADER

def data_loader(ROOT_PATH,BATCH_SIZE=8):
    # ROOT_PATH = '~YaML/data/Ottawa-Dataset/'
    
    DATASET={}
    LOADER={}
    
    _size = 224, 224
    
    # set your transforms
    train_transforms = transforms.Compose([
                               transforms.Resize(_size, interpolation=0),
                               transforms.RandomRotation(180),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(_size, padding = 10), # needed after rotation (with original size)
                           ])
    
    test_transforms = transforms.Compose([
                            
                               transforms.Resize(_size, interpolation=0),
                           ])
    
    trainset = Dataset(root_path = ROOT_PATH, mode = 'train', transforms = train_transforms)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
    
    validset = Dataset(root_path = ROOT_PATH, mode = 'valid', transforms = test_transforms)
    valid_loader = torch.utils.data.DataLoader(validset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 0)
    
    testset = Dataset(root_path = ROOT_PATH, mode = 'test', transforms = test_transforms)
    test_loader = torch.utils.data.DataLoader(testset, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    
    DATASET={'train': trainset,'valid': validset, 'test' : testset}
    LOADER={'train': train_loader,'valid' : valid_loader, 'test' : test_loader}
    
    return DATASET, LOADER
