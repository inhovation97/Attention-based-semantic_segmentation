import os
import cv2
import sys
import glob
import random
import argparse
import logging
from random import shuffle
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.utils.data import DataLoader, random_split
import torchvision
from torchvision import transforms, datasets
import torchvision.transforms.functional as TF

from Codes.data_loader import data_loader
from Codes.metrics import  DiceCELoss ,dice_score ,iou ,dice_pytorch_eval, iou_pytorch_eval, DiceBCELoss
from Codes.ploting import score_plot
from Models.UNet.unet_model import UNet
from Models.UNet.unet_parts import *

from Models.DeepLabV3.deeplabv3_resnet101 import deeplabv3_resnet101
from Models.FCN.fcn_resnet101 import fcn_resnet101

from Models.UNet.unet_model_cbam import UNet_cbam

import copy
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
      torch.cuda.manual_seed_all(42)


# ----------------------------------------------------------------------------------------

def test_scores(model,n_classes,data_path):
    # data_loader
    DATASET, LOADER = data_loader(ROOT_PATH=data_path, BATCH_SIZE=8)
    # root_path : '/home/sh/lab/YaML/'
    # data_path : 'data/Kvasir-SEG/'

    testset=DATASET['test']
    test_loader=LOADER['test']


    # -----------------------------------------------------------------------------------------------------------------------------------

    # train
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-4, weight_decay = 1e-8)
    if n_classes == 1:
        criterion = DiceBCELoss()
    else:
        criterion = DiceCELoss()

    #DiceBCELoss의 bce는 sigmoid를 사용하지 softmax를 사용하지 않으므로 
    #ce구할때 softmax를 사용을 원하면 torch.nn.functional.cross_entropy 를 사용해야한다


    # 5. Begin training
    test_loss_lst, test_dice_lst, test_iou_lst = [], [], []

    # test   
    model.eval()
    test_loss, test_dice, test_iou, test_step = 0, 0, 0, 0
    for image, mask_true_ in test_loader:

        # move images and labels to correct device and type
        image = image.to(device=device)#, dtype=torch.float32, memory_format=torch.channels_last)
        mask_true_ = mask_true_.to(device=device)#, dtype=torch.long)


        if n_classes < 2 :
            mask_true = mask_true_
        else:   # 1-tensor로 반전시킨 텐서 만들고 겹치쳐서 input과 target의 차원 맞춤 
            mask_true = torch.stack([1-mask_true_,mask_true_], dim=1).squeeze(2)

        # predict the mask
        mask_pred = model(image)

        loss = criterion(mask_pred, mask_true)
        test_loss += loss.item()
        if n_classes==1:
            test_iou += iou_pytorch_eval(mask_pred, mask_true).item()
            test_dice += dice_pytorch_eval(mask_pred, mask_true).item()

        else:
            test_iou += iou(mask_pred, mask_true).item()
            test_dice += dice_score(mask_pred, mask_true).item()
        test_step += 1

    # -----------------------------------------------------------------------------------------------------

    test_dice = round( test_dice/test_step,  3 ) 
    test_iou = round( test_iou/test_step, 3 )
    test_loss = round( test_loss/test_step, 3 )

    print(f' test Dice: {test_dice}, test Iou: {test_iou}, test Loss: {test_loss}\n')



    # 리스트 저장
    test_dice_lst.append(test_dice)
    test_loss_lst.append(test_loss)
    test_iou_lst.append(test_iou)
