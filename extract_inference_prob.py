import numpy as np
import os
from PIL import Image as PILImage
import torch
import torchvision
from torchvision import transforms 
import glob
import random
import cv2
from random import shuffle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import torch.nn.functional as F
from torchvision import transforms, datasets
from Codes.dataset import WestChaeVI_Dataset
import copy




def extract_inference_prob(MODE,model,BATCH_SIZE,ROOT_PATH):
    _size = 224, 224
    resize = transforms.Resize(_size, interpolation=0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # set your transforms 
    test_transforms = transforms.Compose([
        transforms.Resize(_size, interpolation=0),
    ])
    # test loader
    data_set = Dataset(root_path = ROOT_PATH, mode = MODE, transforms = test_transforms)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    # [ [BATCH_SIZE , 3, 224, 224], [BATCH_SIZE , 1, 224, 224] ] -> [2, BATCH_SIZE , 3, 224, 224]


    prob_dict = {}
    whole_prob=[]
    incorrect_prob=[]
    correct_prob=[]
    prob_dict['all'] = []
    prob_dict['correct_all'] = []
    prob_dict['incorrect_all'] = []

    plot_input=[]
    #plot_input_to_dic=[]
    
    cnt=0
    
    for i, (img, label) in enumerate(data_loader): #img [BATCH_SIZE , 3, 224, 224] / label [BATCH_SIZE , 1, 224, 224]
        img = img.to(device) 
        label = label.to(device)
        raw_img=img.to('cpu')
        raw_label=label.to('cpu')
    

        pred_mask = model(img) # pred_mask [BATCH_SIZE, 1, 224, 224]
        
        with torch.no_grad():
            if isinstance(pred_mask, dict):
                prob_mask = torch.sigmoid(pred_mask['out']).to('cpu').numpy()
            else:
                prob_mask = torch.sigmoid(pred_mask).to('cpu').numpy()
        # prob_mask 시그모이드 후 numpy 변환 (확률값) -> 224 * 224(확률값) [BATCH_SIZE , 1, 224, 224]

    
        labeled_mask = prob_mask.copy() # labeled_mask [BATCH_SIZE , 1, 224, 224]
        labeled_mask[labeled_mask <= 0.5] = 0.
        labeled_mask[labeled_mask > 0.5] = 1. # 차원 : (BATCH_SIZE, 1, 224, 224) # labeled_mask 확률값의 라벨화 0, 1 으로 변환
    
        # 틀린 부분의 분포만 확인하기
        incorrect_mask = copy.deepcopy(prob_mask) # incorrect_mask [BATCH_SIZE , 1, 224, 224]
        incorrect_mask = np.array(incorrect_mask).transpose(0,2,3,1) # (8, 224, 224)


    #배치 묶음 풀기
    
        raw_img = np.array(raw_img).transpose(0,2,3,1)
        labeled_mask = np.array(labeled_mask).transpose(0,2,3,1) # (8 224 224 1)    0 or 1
        prob_mask = np.array(prob_mask).transpose(0,2,3,1) # (8 224 224 1) # 0~1 probability
        raw_label = np.array(raw_label).transpose(0,2,3,1) # (8 224 224 1)   # ground_Truth


        # Definition inc_mask 
        inc_mask=copy.deepcopy(labeled_mask)
        inc_mask[raw_label != inc_mask] = 2 
        inc_mask[inc_mask != 2] = 0
        inc_mask[inc_mask == 2] = 1


        # Definition inc_labeled_mask
        inc_labeled_mask = copy.deepcopy(labeled_mask)
        inc_labeled_mask[inc_mask == 1] = 100
        inc_labeled_mask[inc_labeled_mask == 1]  = 255
        
        #gradation gra_mask
        gra_mask=copy.deepcopy(prob_mask)
        gra_mask=2*abs(gra_mask-0.5)#0~1사이의 실수값(확률)

    
        n = labeled_mask.shape[0]
        for ii in np.arange(n):
            for pred, label_, prob in zip(labeled_mask[ii].flatten(), raw_label[ii].flatten(), prob_mask[ii].flatten()):
                whole_prob.append(prob)
                if pred != label_:
                    incorrect_prob.append(prob)
                else:
                    correct_prob.append(prob)
      
            plot_input.append([raw_img[ii] , raw_label[ii] , labeled_mask[ii] , inc_labeled_mask[ii] , inc_mask[ii] , gra_mask[ii]]) 
            #raw_img / raw_label / labeled_mask / inc_labeled_mask / inc_mask : 틀린 부위만 1로 나머지 0
            #plot_input_to_dic.append([raw_img[ii] , raw_label[ii] , labeled_mask[ii] , inc_labeled_mask[ii] , inc_mask[ii]])
            prob_dict[f'{MODE}_{cnt}'] = [whole_prob, correct_prob, incorrect_prob]
            
            cnt+=1
            #plot_input_to_dic=[]
      
            prob_dict['all'] += whole_prob
            prob_dict['correct_all'] += correct_prob
            prob_dict['incorrect_all'] += incorrect_prob
      
            whole_prob=[]
            incorrect_prob=[]
            correct_prob=[]
        
    return prob_dict , plot_input


#-----------------------------------------------------------------------------------------------------------------------------------------------------


def extract_inference_prob_n_label(MODE,model,BATCH_SIZE,ROOT_PATH):
    _size = 224, 224
    resize = transforms.Resize(_size, interpolation=0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # set your transforms 
    test_transforms = transforms.Compose([
        transforms.Resize(_size, interpolation=0),
    ])
    # test loader
    data_set = WestChaeVI_Dataset(root_path = ROOT_PATH, mode = MODE, transforms = test_transforms)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    # [ [BATCH_SIZE , 3, 224, 224], [BATCH_SIZE , 1, 224, 224] ] -> [2, BATCH_SIZE , 3, 224, 224]


    prob_dict = {}
    whole_prob=[]
    incorrect_prob=[]
    correct_prob=[]
    
    whole_label=[]
    incorrect_label=[]
    correct_label=[]
    
    prob_dict['all'] = [[],[]] #prob_dict['all'][0]에 확률값  prob_dict['all'][1]에 라벨값
    prob_dict['correct_all'] = [[],[]]
    prob_dict['incorrect_all'] = [[],[]]

    plot_input=[]
    #plot_input_to_dic=[]
    
    cnt=0
    
    for i, (img, label) in enumerate(data_loader): #img [BATCH_SIZE , 3, 224, 224] / label [BATCH_SIZE , 1, 224, 224]
        img = img.to(device) 
        label = label.to(device)
        raw_img=img.to('cpu')
        raw_label=label.to('cpu')
    

        pred_mask = model(img) # pred_mask [BATCH_SIZE, 1, 224, 224]
        
        with torch.no_grad():
            if isinstance(pred_mask, dict):
                prob_mask = torch.sigmoid(pred_mask['out']).to('cpu').numpy()
            else:
                prob_mask = torch.sigmoid(pred_mask).to('cpu').numpy()
        # prob_mask 시그모이드 후 numpy 변환 (확률값) -> 224 * 224(확률값) [BATCH_SIZE , 1, 224, 224]

    
        labeled_mask = copy.deepcopy(prob_mask) # labeled_mask [BATCH_SIZE , 1, 224, 224]
        labeled_mask[labeled_mask <= 0.5] = 0.
        labeled_mask[labeled_mask > 0.5] = 1. # 차원 : (BATCH_SIZE, 1, 224, 224) # labeled_mask 확률값의 라벨화 0, 1 으로 변환
    
        # 틀린 부분의 분포만 확인하기
        incorrect_mask = copy.deepcopy(prob_mask) # incorrect_mask [BATCH_SIZE , 1, 224, 224]
        incorrect_mask = np.array(incorrect_mask).transpose(0,2,3,1) # (8, 224, 224)


    #배치 묶음 풀기
    
        raw_img = np.array(raw_img).transpose(0,2,3,1)
        labeled_mask = np.array(labeled_mask).transpose(0,2,3,1) # (8 224 224 1)    0 or 1
        prob_mask = np.array(prob_mask).transpose(0,2,3,1) # (8 224 224 1) # 0~1 probability
        raw_label = np.array(raw_label).transpose(0,2,3,1) # (8 224 224 1)   # ground_Truth


        # Definition inc_mask 
        inc_mask=copy.deepcopy(labeled_mask)
        inc_mask[raw_label != inc_mask] = 2 
        inc_mask[inc_mask != 2] = 0
        inc_mask[inc_mask == 2] = 1


        # Definition inc_labeled_mask
        inc_labeled_mask = copy.deepcopy(labeled_mask)
        inc_labeled_mask[inc_mask == 1] = 100
        inc_labeled_mask[inc_labeled_mask == 1]  = 255
        
        #gradation gra_mask
        gra_mask=copy.deepcopy(prob_mask)
        gra_mask=2*abs(gra_mask-0.5)#0~1사이의 실수값(확률)

    
        n = labeled_mask.shape[0]
        for ii in np.arange(n):
            for pred, label_, prob in zip(labeled_mask[ii].flatten(), raw_label[ii].flatten(), prob_mask[ii].flatten()):
                whole_prob.append(prob)
                whole_label.append(label_)
                if pred != label_:
                    incorrect_prob.append(prob)
                    incorrect_label.append(label_)
                else:
                    correct_prob.append(prob)
                    correct_label.append(label_)
      
            plot_input.append([raw_img[ii] , raw_label[ii] , labeled_mask[ii] , inc_labeled_mask[ii] , inc_mask[ii] , gra_mask[ii]]) 
            #raw_img / raw_label / labeled_mask / inc_labeled_mask / inc_mask : 틀린 부위만 1로 나머지 0
            #plot_input_to_dic.append([raw_img[ii] , raw_label[ii] , labeled_mask[ii] , inc_labeled_mask[ii] , inc_mask[ii]])
            prob_dict[f'{MODE}_{cnt}'] = [whole_prob, correct_prob, incorrect_prob]
            
            cnt+=1
            #plot_input_to_dic=[]
            
            prob_dict['all'][0]+=whole_prob
            prob_dict['all'][1]+=whole_label
            
            prob_dict['incorrect_all'][0]+=incorrect_prob
            prob_dict['incorrect_all'][1]+=incorrect_label
            
            prob_dict['correct_all'][0]+=correct_prob
            prob_dict['correct_all'][1]+=correct_label
      
            whole_prob=[]
            incorrect_prob=[]
            correct_prob=[]

            whole_label=[]
            incorrect_label=[]
            correct_label=[]
        
    return prob_dict , plot_input

def extract_inference_prob_n_label_softmax(MODE,model,BATCH_SIZE,ROOT_PATH):
    _size = 224, 224
    resize = transforms.Resize(_size, interpolation=0)
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # set your transforms 
    test_transforms = transforms.Compose([
        transforms.Resize(_size, interpolation=0),
    ])
    # test loader
    data_set = WestChaeVI_Dataset(root_path = ROOT_PATH, mode = MODE, transforms = test_transforms)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size = BATCH_SIZE, shuffle = False, num_workers = 0)
    # [ [BATCH_SIZE , 3, 224, 224], [BATCH_SIZE , 1, 224, 224] ] -> [2, BATCH_SIZE , 3, 224, 224]


    prob_dict = {}
    whole_prob=[]
    incorrect_prob=[]
    correct_prob=[]
    
    whole_label=[]
    incorrect_label=[]
    correct_label=[]
    
    prob_dict['all'] = [[],[]] #prob_dict['all'][0]에 확률값  prob_dict['all'][1]에 라벨값
    prob_dict['correct_all'] = [[],[]]
    prob_dict['incorrect_all'] = [[],[]]

    plot_input=[]
    #plot_input_to_dic=[]
    
    cnt=0
    
    for i, (img, label) in enumerate(data_loader): #img [BATCH_SIZE , 3, 224, 224] / label [BATCH_SIZE , 1, 224, 224]
        img = img.to(device) 
        label = label.to(device)
        raw_img=img.to('cpu')
        raw_label=label.to('cpu')
    

        pred_mask = model(img) # pred_mask [BATCH_SIZE, 1, 224, 224]

        with torch.no_grad():

            # dict형태로 데이터가 들어오는 경우가 있음 
            if isinstance(pred_mask, dict):
                pred_mask = pred_mask['out']

            prob_mask = F.softmax(pred_mask, dim=1)
            labeled_mask = torch.argmax(prob_mask, dim=1).unsqueeze(1).to('cpu').numpy()


        # prob_mask 시그모이드 후 numpy 변환 (확률값) -> 224 * 224(확률값) [BATCH_SIZE , 1, 224, 224]
        prob_mask=prob_mask[:,1,:,:].unsqueeze(1).contiguous().to('cpu').numpy()
        # 틀린 부분의 분포만 확인하기
        
        incorrect_mask = copy.deepcopy(prob_mask) # incorrect_mask [BATCH_SIZE , 1, 224, 224]
        incorrect_mask = np.array(incorrect_mask).transpose(0,2,3,1) # (8, 224, 224)


    #배치 묶음 풀기
    
        raw_img = np.array(raw_img).transpose(0,2,3,1)
        labeled_mask = np.array(labeled_mask).transpose(0,2,3,1) # (8 224 224 1)    0 or 1
        prob_mask = np.array(prob_mask).transpose(0,2,3,1) # (8 224 224 1) # 0~1 probability
        raw_label = np.array(raw_label).transpose(0,2,3,1) # (8 224 224 1)   # ground_Truth


        # Definition inc_mask 
        inc_mask=copy.deepcopy(labeled_mask)
        inc_mask[raw_label != inc_mask] = 2 
        inc_mask[inc_mask != 2] = 0
        inc_mask[inc_mask == 2] = 1


        # Definition inc_labeled_mask
        inc_labeled_mask = copy.deepcopy(labeled_mask)
        inc_labeled_mask[inc_mask == 1] = 100
        inc_labeled_mask[inc_labeled_mask == 1]  = 255
        
        #gradation gra_mask
        gra_mask=copy.deepcopy(prob_mask)
        gra_mask=2*abs(gra_mask-0.5)#0~1사이의 실수값(확률)

    
        n = labeled_mask.shape[0]
        for ii in np.arange(n):
            for pred, label_, prob in zip(labeled_mask[ii].flatten(), raw_label[ii].flatten(), prob_mask[ii].flatten()):
                whole_prob.append(prob)
                whole_label.append(label_)
                if pred != label_:
                    incorrect_prob.append(prob)
                    incorrect_label.append(label_)
                else:
                    correct_prob.append(prob)
                    correct_label.append(label_)
      
            plot_input.append([raw_img[ii] , raw_label[ii] , labeled_mask[ii] , inc_labeled_mask[ii] , inc_mask[ii] , gra_mask[ii]]) 
            #raw_img / raw_label / labeled_mask / inc_labeled_mask / inc_mask : 틀린 부위만 1로 나머지 0
            #plot_input_to_dic.append([raw_img[ii] , raw_label[ii] , labeled_mask[ii] , inc_labeled_mask[ii] , inc_mask[ii]])
            prob_dict[f'{MODE}_{cnt}'] = [whole_prob, correct_prob, incorrect_prob]
            
            cnt+=1
            #plot_input_to_dic=[]
            
            prob_dict['all'][0]+=whole_prob
            prob_dict['all'][1]+=whole_label
            
            prob_dict['incorrect_all'][0]+=incorrect_prob
            prob_dict['incorrect_all'][1]+=incorrect_label
            
            prob_dict['correct_all'][0]+=correct_prob
            prob_dict['correct_all'][1]+=correct_label
      
            whole_prob=[]
            incorrect_prob=[]
            correct_prob=[]

            whole_label=[]
            incorrect_label=[]
            correct_label=[]
        
    return prob_dict , plot_input


