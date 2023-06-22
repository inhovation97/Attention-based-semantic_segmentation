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

from Codes.data_loader import data_loader,data_loader_resized
from Codes.metrics import  DiceCELoss ,dice_score ,iou ,dice_pytorch_eval, iou_pytorch_eval, DiceBCELoss
from Codes.ploting import score_plot
from Models.UNet.unet_model import UNet
from Models.UNet.unet_parts import *

from Models.DeepLabV3.deeplabv3_resnet101 import deeplabv3_resnet101
from Models.FCN.fcn_resnet101 import fcn_resnet101

from Models.UNet.unet_model_cbam import UNet_cbam


from Models.FCBFormer.models import FCBFormer

from Models.FCBFormer.models_cbam import FCBFormer_cbam

import copy
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
      torch.cuda.manual_seed_all(42)


# ----------------------------------------------------------------------------------------

def train_model(model_name, mode, root_path, data_path, n_classes, n_channels=3, batch_size=8, epoches=150, att=False, act_types=['sigmoid'],use_selu=False,alpha=1.0):
# root_path = '/home/sh/lab/YaML/'
    ## import model & fine_tuning
    if data_path.split('/')[-1] == '':
        data_name=data_path.split('/')[-2]
    else:
        data_name=data_path.split('/')[-1]

        
    if model_name == 'unet':
        
        if att == 'cbam':        
                        
            model = UNet_cbam(n_channels=n_channels, n_classes= n_classes, mode_= mode, reduction_ratio=16, act_types=act_types,use_selu=use_selu, alpha=alpha).to(device)

            weights_path =  root_path + 'Models/UNet/unet_carvana_scale0.5_epoch2.pth'
            weights = torch.load(weights_path)
            if n_classes==1:
                del weights['outc.conv.weight']
                del weights['outc.conv.bias']
        
            model.load_state_dict(weights,strict=False)
            print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} (alpha : {alpha}) / Spetial Activation : {act_types}\nEpoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")

        else:
            model = UNet(n_channels=n_channels, n_classes= n_classes, mode_= mode, att = att).to(device)

            weights_path =  root_path + 'Models/UNet/unet_carvana_scale0.5_epoch2.pth'
            weights = torch.load(weights_path)
            if n_classes==1:
                del weights['outc.conv.weight']
                del weights['outc.conv.bias']

            model.load_state_dict(weights,strict=False)
            print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} (alpha : {alpha})\nepoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")


    elif model_name == 'deeplabv3':
        model=deeplabv3_resnet101(outchannels=n_classes, mode_=mode, att=att, act_types=act_types, use_selu=use_selu, alpha=alpha).to(device)
        
        print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} (alpha : {alpha}) / Spetial Activation : {act_types}\nepoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")
        
        
    elif model_name == 'fcn':
        model = fcn_resnet101(outchannels= n_classes, mode_= mode, att = att, act_types=act_types, use_selu=use_selu, alpha=alpha).to(device)

        print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} (alpha : {alpha}) / Spetial Activation : {act_types}\nepoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")

        
        
       
    elif model_name == 'fcb': 
        
        if att == 'cbam':
            
            model = FCBFormer_cbam(size=224, n_classes = n_classes, mode_= mode, act_types=act_types, reduction_ratio=16 , alpha=alpha).to(device)
        
            print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} (alpha : {alpha}) / Spetial Activation : {act_types}\nEpoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")
        
        
        else:
            model = FCBFormer(size=224, n_classes = n_classes, mode_= mode).to(device)
        
            print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} (alpha : {alpha})/ Spetial Activation : {act_types}\nEpoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")
        
        
        
        
    
    # data_loader
    if data_name == 'ISIC':
        DATASET, LOADER = data_loader_resized(ROOT_PATH=root_path + data_path, BATCH_SIZE=batch_size)
    else:
        DATASET, LOADER = data_loader(ROOT_PATH=root_path + data_path, BATCH_SIZE=batch_size)
    # root_path : '/home/sh/lab/YaML/'
    # data_path : 'data/Kvasir-SEG/'

    trainset=DATASET['train']
    validset=DATASET['valid']
    testset=DATASET['test']

    train_loader=LOADER['train']
    valid_loader=LOADER['valid']
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
    train_loss_lst, train_dice_lst, train_iou_lst, valid_loss_lst, valid_dice_lst, valid_iou_lst = [], [], [], [], [], []
    state={}
    state_st = {}
    state200 = {}
    earlystop=True



    for epoch in range(1, epoches + 1):
        train_loss, train_dice, train_iou, train_step = 0, 0, 0, 0
        
        
    # Train
        model.train()
        for images, true_masks_ in train_loader:  # batch_size 묶음으로 나옴 (8묶음으로 62세트 학습)
            
            images = images.to(device=device) #, dtype=torch.float32, memory_format = torch.channels_last)
            true_masks_ = true_masks_.to(device=device) #, dtype=torch.long)
        
        
            if n_classes < 2 :
                true_masks = true_masks_
            else:   # 1-tensor로 반전시킨 텐서 만들고 겹치쳐서 input과 target의 차원 맞춤
                true_masks = torch.stack([1-true_masks_,true_masks_], dim=1).squeeze(2)
        
            
        
            pred = model(images) # 1 2 224 224

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(pred, true_masks)
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            if n_classes==1:
                train_iou += iou_pytorch_eval(pred, true_masks).item()
                train_dice += dice_pytorch_eval(pred, true_masks).item()

            else:
                train_iou += iou(pred, true_masks).item()
                train_dice += dice_score(pred, true_masks).item()
            train_step += 1

    # Validate   
        model.eval()
        val_loss, val_dice, val_iou, val_step = 0, 0, 0, 0
        for image, mask_true_ in valid_loader:

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
            val_loss += loss.item()
            if n_classes==1:
                val_iou += iou_pytorch_eval(mask_pred, mask_true).item()
                val_dice += dice_pytorch_eval(mask_pred, mask_true).item()

            else:
                val_iou += iou(mask_pred, mask_true).item()
                val_dice += dice_score(mask_pred, mask_true).item()
            val_step += 1

# -----------------------------------------------------------------------------------------------------

        # epoch마다 프린팅
        Train_Dice = round( train_dice/train_step, 3 ) 
        Train_Iou = round( train_iou/train_step, 3 )
        Train_Loss = round( train_loss/train_step, 3 )

        val_dice = round( val_dice/val_step,  3 )  # val_step = 21 (0~20)
        val_iou = round( val_iou/val_step, 3 )
        val_loss = round( val_loss/val_step, 3 )

        print(f'Epoch: {epoch}/{epoches}, Train Dice: {Train_Dice}, Train Iou: {Train_Iou}, Train Loss: {Train_Loss}')
        print(f'Epoch: {epoch}/{epoches}, valid Dice: {val_dice}, valid Iou: {val_iou}, valid Loss: {val_loss}\n')
        
        


        # 리스트 저장
        train_dice_lst.append(Train_Dice)
        train_loss_lst.append(Train_Loss)
        train_iou_lst.append(Train_Iou)
    
        valid_dice_lst.append(val_dice)
        valid_loss_lst.append(val_loss)
        valid_iou_lst.append(val_iou)
    

  
    
    
        # 모델 저장 (best epoch)
        if (epoch <= 200) and (np.max(valid_dice_lst) <= val_dice):
            state200['epoch'] = epoch
            state200['net'] = copy.deepcopy(model.state_dict())
            
            state200['train_dice'] = Train_Dice
            state200['val_dice'] = val_dice

            state200['train_iou'] = Train_Iou
            state200['val_iou'] = val_iou

            state200['train_loss'] = Train_Loss
            state200['val_loss'] = val_loss
            
             
            
        if np.max(valid_dice_lst) <= val_dice:
            state['epoch'] = epoch
            state['net'] = copy.deepcopy(model.state_dict())
            
            state['train_dice'] = Train_Dice
            state['val_dice'] = val_dice

            state['train_iou'] = Train_Iou
            state['val_iou'] = val_iou

            state['train_loss'] = Train_Loss
            state['val_loss'] = val_loss

           
            
            if earlystop:    
                state_st['epoch'] = epoch
                state_st['net'] = copy.deepcopy(model.state_dict())
            
                state_st['train_dice'] = Train_Dice
                state_st['val_dice'] = val_dice

                state_st['train_iou'] = Train_Iou
                state_st['val_iou'] = val_iou

                state_st['train_loss'] = Train_Loss
                state_st['val_loss'] = val_loss
            
        if ((epoch-state['epoch'])>30 and (epoch > 200)):
            earlystop=False              
            
            

        if ( (epoch-state['epoch'])>30 and (epoch > 200) and earlystop ) or ((epoch==epoches) and earlystop):
            print("# 200에포크 이후 30 epoch동안 Best epoch가 갱신되지 않아 가중치를 저장합니다 #")

            os.makedirs(root_path+'check_points/'+data_name+'/Basic/earlystop' , exist_ok=True)
            os.makedirs(root_path+'check_points/'+data_name+'/CBAM/earlystop' , exist_ok=True)
            earlystop=False
            
            
            if state200['val_dice']<=val_dice:
            
                earlystop_epoch = copy.deepcopy(epoch)
                earlystop_val_dice = copy.deepcopy(val_dice)
                earlystop_val_iou = copy.deepcopy(val_iou)
                earlystop_val_loss = copy.deepcopy(val_loss)
            
            else:
                
                
                earlystop_epoch = copy.deepcopy(state200['epoch'])
                earlystop_val_dice = copy.deepcopy(state200['val_dice'])
                earlystop_val_iou = copy.deepcopy(state200['val_iou'])
                earlystop_val_loss = copy.deepcopy(state200['val_loss'])
                

            if att == 'cbam':
                if act_types==['sigmoid']:
                    torch.save(state, root_path + 'check_points/{}/CBAM/earlystop/{}_{}_S{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state['epoch'], state['val_dice']))
                elif act_types==['tanh']:
                    torch.save(state, root_path + 'check_points/{}/CBAM/earlystop/{}_{}_T{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state['epoch'], state['val_dice']))
                elif act_types==['sigmoid','tanh'] or act_types==['tanh','sigmoid']:
                    torch.save(state, root_path + 'check_points/{}/CBAM/earlystop/{}_{}_ST{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state['epoch'], state['val_dice']))


            else:
                if att:
                    torch.save(state, root_path + 'check_points/{}/Att/earlystop/{}_{}_{}_{}_{}.pth'.format(data_name,mode, 'Att_'+model_name,data_name+'_best', state['epoch'], state['val_dice']))

                elif not att:
                    if n_classes==1:
                        torch.save(state, root_path + 'check_points/{}/Basic/earlystop/{}1_{}_{}_{}_{}.pth'.format(data_name,mode, model_name,data_name+'_best', state['epoch'], state['val_dice']))

                    else:    
                        torch.save(state, root_path + 'check_points/{}/Basic/earlystop/{}_{}_{}_{}_{}.pth'.format(data_name, mode, model_name, data_name+'_best', state['epoch'], state['val_dice']))



            
    os.makedirs(root_path+'check_points/'+data_name+'/Basic/200' , exist_ok=True)
    os.makedirs(root_path+'check_points/'+data_name+'/Att/200' , exist_ok=True)
    os.makedirs(root_path+'check_points/'+data_name+'/CBAM/200' , exist_ok=True)
    os.makedirs(root_path+'check_points/'+data_name+'/Basic/300' , exist_ok=True)
    os.makedirs(root_path+'check_points/'+data_name+'/Att/300' , exist_ok=True)
    os.makedirs(root_path+'check_points/'+data_name+'/CBAM/300' , exist_ok=True)
    os.makedirs(root_path+'check_points/'+data_name+'/Basic/earlystop' , exist_ok=True)
    os.makedirs(root_path+'check_points/'+data_name+'/CBAM/earlystop' , exist_ok=True)

            
            
    if att == 'cbam':
        if act_types==['sigmoid']:
            torch.save(state, root_path + 'check_points/{}/CBAM/300/{}_{}_S{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state['epoch'], state['val_dice']))
        elif act_types==['tanh']:
            torch.save(state, root_path + 'check_points/{}/CBAM/300/{}_{}_T{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state['epoch'], state['val_dice']))
        elif act_types==['sigmoid','tanh'] or act_types==['tanh','sigmoid']:
            torch.save(state, root_path + 'check_points/{}/CBAM/300/{}_{}_ST{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state['epoch'], state['val_dice']))
        
        
    else:
        if att:
            torch.save(state, root_path + 'check_points/{}/Att/300/{}_{}_{}_{}_{}.pth'.format(data_name,mode, 'Att_'+model_name,data_name+'_best', state['epoch'], state['val_dice']))
    
        elif not att:
            if n_classes==1:
                torch.save(state, root_path + 'check_points/{}/Basic/300/{}1_{}_{}_{}_{}.pth'.format(data_name,mode, model_name,data_name+'_best', state['epoch'], state['val_dice']))
        
            else:    
                torch.save(state, root_path + 'check_points/{}/Basic/300/{}_{}_{}_{}_{}.pth'.format(data_name, mode, model_name, data_name+'_best', state['epoch'], state['val_dice']))

    if att == 'cbam':
        if act_types==['sigmoid']:
            torch.save(state200, root_path + 'check_points/{}/CBAM/200/{}_{}_S{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state200['epoch'], state200['val_dice']))
        elif act_types==['tanh']:
            torch.save(state200, root_path + 'check_points/{}/CBAM/200/{}_{}_T{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state200['epoch'], state200['val_dice']))
        elif act_types==['sigmoid','tanh'] or act_types==['tanh','sigmoid']:
            torch.save(state200, root_path + 'check_points/{}/CBAM/200/{}_{}_ST{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state200['epoch'], state200['val_dice']))
        
        
    else:
        if att:
            torch.save(state200, root_path + 'check_points/{}/Att/200/{}_{}_{}_{}_{}.pth'.format(data_name,mode, 'Att_'+model_name,data_name+'_best', state200['epoch'], state200['val_dice']))
    
        elif not att:
            if n_classes==1:
                torch.save(state200, root_path + 'check_points/{}/Basic/200/{}1_{}_{}_{}_{}.pth'.format(data_name,mode, model_name,data_name+'_best', state200['epoch'], state200['val_dice']))
        
            else:    
                torch.save(state200, root_path + 'check_points/{}/Basic/200/{}_{}_{}_{}_{}.pth'.format(data_name, mode, model_name, data_name+'_best', state200['epoch'], state200['val_dice']))  

    if att == 'cbam':
        if act_types==['sigmoid']:
            torch.save(state_st, root_path + 'check_points/{}/CBAM/earlystop/{}_{}_S{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state_st['epoch'], state_st['val_dice']))
        elif act_types==['tanh']:
            torch.save(state_st, root_path + 'check_points/{}/CBAM/earlystop/{}_{}_T{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state_st['epoch'], state_st['val_dice']))
        elif act_types==['sigmoid','tanh'] or act_types==['tanh','sigmoid']:
            torch.save(state_st, root_path + 'check_points/{}/CBAM/earlystop/{}_{}_ST{}_{}_{}_{}_{}.pth'.format(data_name, mode, att, alpha, model_name, data_name+'_best', state_st['epoch'], state_st['val_dice']))
        
        
    else:
        if att:
            torch.save(state_st, root_path + 'check_points/{}/Att/earlystop/{}_{}_{}_{}_{}.pth'.format(data_name,mode, 'Att_'+model_name,data_name+'_best', state_st['epoch'], state_st['val_dice']))
    
        elif not att:
            if n_classes==1:
                torch.save(state_st, root_path + 'check_points/{}/Basic/earlystop/{}1_{}_{}_{}_{}.pth'.format(data_name,mode, model_name,data_name+'_best', state_st['epoch'], state_st['val_dice']))
        
            else:    
                torch.save(state_st, root_path + 'check_points/{}/Basic/earlystop/{}_{}_{}_{}_{}.pth'.format(data_name, mode, model_name, data_name+'_best', state_st['epoch'], state_st['val_dice']))  
                

    
    
    print('200 until best epoch : {} / dice_score : {} / Iou : {} / Loss : {}'.format(state200['epoch'], state200['val_dice'],state200['val_iou'],state200['val_loss']))
    print('Early Stopping best epoch : {} / dice_score : {} / Iou : {} / Loss : {}'.format(state_st['epoch'], state_st['val_dice'],state_st['val_iou'],state_st['val_loss']))
    print('total best epoch : {} / dice_score : {} / Iou : {} / Loss : {}'.format(state['epoch'], state['val_dice'],state['val_iou'],state['val_loss']))
    
    score_plot(epoch,state,train_dice_lst,train_iou_lst,train_loss_lst,valid_dice_lst,valid_iou_lst,valid_loss_lst)
    

    
    
    
def train_continue_model(weights, model_name, mode, root_path, data_path, n_classes,  n_channels=3, batch_size=8, epoches=150, att=False, act_types=['sigmoid'],use_selu=False ):
# root_path = '/home/sh/lab/YaML/'
    ## import model & fine_tuning
    if data_path.split('/')[-1] == '':
        data_name=data_path.split('/')[-2]
    else:
        data_name=data_path.split('/')[-1]
        

        
    if model_name == 'unet':
        
        if att == 'cbam':        
                        
            model = UNet_cbam(n_channels=n_channels, n_classes= n_classes, mode_= mode, reduction_ratio=16, act_types=act_types,use_selu=use_selu).to(device)

        
            model.load_state_dict(weights['net'])
            print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} / Spetial Activation : {act_types}\nEpoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")

        else:
            model = UNet(n_channels=n_channels, n_classes= n_classes, mode_= mode, att = att).to(device)

            model.load_state_dict(weights['net'])
            print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att}\nepoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")


    elif model_name == 'deeplabv3':
        model=deeplabv3_resnet101(outchannels=n_classes, mode_=mode, att=att, act_types=act_types, use_selu=use_selu).to(device)
        
        model.load_state_dict(weights['net'])
        print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} / Spetial Activation : {act_types}\nepoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")
        
        
    elif model_name == 'fcn':
        model = fcn_resnet101(outchannels= n_classes, mode_= mode, att = att, act_types=act_types, use_selu=use_selu).to(device)
        
        model.load_state_dict(weights['net'])
        print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} / Spetial Activation : {act_types}\nepoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")
               
    elif model_name == 'fcb': 
        
        if att == 'cbam':
            
            model = FCBFormer_cbam(size=224, n_classes = n_classes, mode_= mode, act_types=act_types, reduction_ratio=16 )
        
            print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} / Spetial Activation : {act_types}\nEpoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")
        
        
        else:
            model = FCBFormer(size=224, n_classes = n_classes, mode_= mode)
        
            print(f"모델 : {model_name} | 활성화함수: {mode} | Add Attetion : {att} / Spetial Activation : {act_types}\nEpoches : {epoches} | Batch Size : {batch_size} | 이미지 채널수 : {n_channels} | Class 개수 : {n_classes} ")

    
    # data_loader
    DATASET, LOADER = data_loader(ROOT_PATH=root_path + data_path, BATCH_SIZE=batch_size)
    # root_path : '/home/sh/lab/YaML/'
    # data_path : 'data/Kvasir-SEG/'

    trainset=DATASET['train']
    validset=DATASET['valid']
    testset=DATASET['test']

    train_loader=LOADER['train']
    valid_loader=LOADER['valid']
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
    train_loss_lst, train_dice_lst, train_iou_lst, valid_loss_lst, valid_dice_lst, valid_iou_lst = [], [], [], [], [], []
    state={}
   
    start_epoch=copy.deepcopy(weights['epoch'])
    
    state['epoch'] = weights['epoch']
    state['net'] = weights['net'] 

    state['train_dice'] =weights['train_dice']
    state['val_dice'] = weights['val_dice'] 

    state['train_iou'] = weights['train_iou'] 
    state['val_iou'] = weights['val_iou'] 

    state['train_loss'] = weights['train_loss'] 
    state['val_loss'] = weights['val_loss'] 
    

    train_dice_lst.append(weights['train_dice'])
    train_loss_lst.append(weights['train_loss'])
    train_iou_lst.append(weights['train_iou'])

    valid_dice_lst.append(weights['val_dice'])
    valid_loss_lst.append(weights['val_loss'])
    valid_iou_lst.append(weights['val_iou'] )
        




    for epoch in range(weights['epoch'], epoches + 1):     # ( 가져온 가중치에서 학습한 에포크부터 목표 에포크까지 )    
        train_loss, train_dice, train_iou, train_step = 0, 0, 0, 0
        
    # Train
        model.train()
        for images, true_masks_ in train_loader:  # batch_size 묶음으로 나옴 (8묶음으로 62세트 학습)
            
            images = images.to(device=device) #, dtype=torch.float32, memory_format = torch.channels_last)
            true_masks_ = true_masks_.to(device=device) #, dtype=torch.long)
        
        
            if n_classes < 2 :
                true_masks = true_masks_
            else:   # 1-tensor로 반전시킨 텐서 만들고 겹치쳐서 input과 target의 차원 맞춤
                true_masks = torch.stack([1-true_masks_,true_masks_], dim=1).squeeze(2)
        
            
        
            pred = model(images) # 1 2 224 224

            optimizer.zero_grad(set_to_none=True)
            loss = criterion(pred, true_masks)
            loss.backward()
            optimizer.step()
        
            train_loss += loss.item()
            if n_classes==1:
                train_iou += iou_pytorch_eval(pred, true_masks).item()
                train_dice += dice_pytorch_eval(pred, true_masks).item()

            else:
                train_iou += iou(pred, true_masks).item()
                train_dice += dice_score(pred, true_masks).item()
            train_step += 1

    # Validate   
        model.eval()
        val_loss, val_dice, val_iou, val_step = 0, 0, 0, 0
        for image, mask_true_ in valid_loader:

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
            val_loss += loss.item()
            if n_classes==1:
                val_iou += iou_pytorch_eval(mask_pred, mask_true).item()
                val_dice += dice_pytorch_eval(mask_pred, mask_true).item()

            else:
                val_iou += iou(mask_pred, mask_true).item()
                val_dice += dice_score(mask_pred, mask_true).item()
            val_step += 1

# -----------------------------------------------------------------------------------------------------

        # epoch마다 프린팅
        Train_Dice = round( train_dice/train_step, 3 ) 
        Train_Iou = round( train_iou/train_step, 3 )
        Train_Loss = round( train_loss/train_step, 3 )

        val_dice = round( val_dice/val_step,  3 )  # val_step = 21 (0~20)
        val_iou = round( val_iou/val_step, 3 )
        val_loss = round( val_loss/val_step, 3 )

        print(f'Epoch: {epoch}/{epoches}, Train Dice: {Train_Dice}, Train Iou: {Train_Iou}, Train Loss: {Train_Loss}')
        print(f'Epoch: {epoch}/{epoches}, valid Dice: {val_dice}, valid Iou: {val_iou}, valid Loss: {val_loss}\n')
        
        


        # 리스트 저장
        train_dice_lst.append(Train_Dice)
        train_loss_lst.append(Train_Loss)
        train_iou_lst.append(Train_Iou)
    
        valid_dice_lst.append(val_dice)
        valid_loss_lst.append(val_loss)
        valid_iou_lst.append(val_iou)
    
        # 모델 저장 (best epoch)

            
        if np.max(valid_dice_lst) <= val_dice:
            state['epoch'] = epoch
            state['net'] = copy.deepcopy(model.state_dict())
            
            state['train_dice'] = Train_Dice
            state['val_dice'] = val_dice

            state['train_iou'] = Train_Iou
            state['val_iou'] = val_iou

            state['train_loss'] = Train_Loss
            state['val_loss'] = val_loss
        
         
            

    os.makedirs(root_path+'check_points/'+data_name+'/Basic/Continue' , exist_ok=True)
    os.makedirs(root_path+'check_points/'+data_name+'/Att/Continue' , exist_ok=True)
    os.makedirs(root_path+'check_points/'+data_name+'/CBAM/Continue' , exist_ok=True)
            
    if att == 'cbam':
        if act_types==['sigmoid']:
            torch.save(state, root_path + 'check_points/{}/CBAM/Continue/{}_{}_S_{}_{}_{}_{}.pth'.format(data_name, mode, att, model_name, data_name+'_best', state['epoch'], state['val_dice']))
        elif act_types==['tanh']:
            torch.save(state, root_path + 'check_points/{}/CBAM/Continue/{}_{}_T_{}_{}_{}_{}.pth'.format(data_name, mode, att, model_name, data_name+'_best', state['epoch'], state['val_dice']))
        elif act_types==['sigmoid','tanh'] or act_types==['tanh','sigmoid']:
            torch.save(state, root_path + 'check_points/{}/CBAM/Continue/{}_{}_ST_{}_{}_{}_{}.pth'.format(data_name, mode, att, model_name, data_name+'_best', state['epoch'], state['val_dice']))
        
        
    else:
        if att:
            torch.save(state, root_path + 'check_points/{}/Att/Continue/{}_{}_{}_{}_{}.pth'.format(data_name,mode, 'Att_'+model_name,data_name+'_best', state['epoch'], state['val_dice']))
    
        elif not att:
            if n_classes==1:
                torch.save(state, root_path + 'check_points/{}/Basic/Continue/{}1_{}_{}_{}_{}.pth'.format(data_name,mode, model_name,data_name+'_best', state['epoch'], state['val_dice']))
        
            else:    
                torch.save(state, root_path + 'check_points/{}/Basic/Continue/{}_{}_{}_{}_{}.pth'.format(data_name, mode, model_name, data_name+'_best', state['epoch'], state['val_dice']))


    
    print('total best epoch : {} / dice_score : {} / Iou : {} / Loss : {}'.format(state['epoch'], state['val_dice'],state['val_iou'],state['val_loss']))
    
    score_plot(epoch,state,train_dice_lst,train_iou_lst,train_loss_lst,valid_dice_lst,valid_iou_lst,valid_loss_lst,start_epoch=(start_epoch-1))