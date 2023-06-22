import glob
import os
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
import random
import time
import torch
import seaborn as sns
import warnings
from Codes.extract_inference_prob import extract_inference_prob_n_label
from Models.Unet import Unet
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(42)
if device == 'cuda':
    torch.cuda.manual_seed_all(42)


warnings.filterwarnings("ignore")

def plot_mode_prediction(model, DEVICE, valid_images, valid_labels):#, save_dir, epochs):
    DEVICE = device
    #model = unet.to(device)
    valid_images = sorted(glob.glob('/home/sh/lab/YaML/data/Kvasir-SEG/test/kv_image/*'))
    valid_labels = sorted(glob.glob('/home/sh/lab/YaML/data/Kvasir-SEG/test/kv_label/*'))

    size = (224, 224)

    i = random.sample(list(np.arange(0, len(valid_images))), 1)
    i = i[0]
    path_img = valid_images[i]
    path_label = valid_labels[i]

    # figure 생성
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    plt.axis('off')

    # eval 전 이미지 전처리
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)

    # label 이미지 전처리
    label = cv2.imread(path_label)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    label = cv2.resize(label, size, interpolation = cv2.INTER_LINEAR)

    model.eval()
    eval_image = img / 255.0
    eval_image = eval_image.astype(np.float32)
    eval_image = eval_image.transpose((2,0,1))
    eval_image = torch.from_numpy(eval_image).unsqueeze(0) # Batch 채널 추가 -> (1, 3, 256, 256)
    eval_image = eval_image.to( device=DEVICE, dtype = torch.float32 )


    # we do not need to calculate gradients
    with torch.no_grad():
        # Prediction
        pred = model(eval_image)  # (1, 2, 224, 224)


    if isinstance(pred, dict):
        pred = pred['out']  # (1 2 224 224)
        
    pred = F.softmax(pred, dim=1)  # (1 2 224 224)

    
    mask = torch.argmax(pred, dim=1) #(1 224 224)

    #with torch.no_grad(): 
    #   mask = torch.argmax( mask, dim=1 )  # argmax 하기전에는 (1, 2, 224, 224)
    mask = mask.squeeze() # (2, 224, 224) 오류 남 (224, 224)가 되야함!


    mask = mask.to(device = 'cpu', dtype = torch.int64).numpy() # tensor to numpy (반드시 디바이스도 변경)
    mask = np.stack( (mask,)*3, axis=-1 ) # (224,224,3)

    # 마스킹을 보여주기 위해 흰색처리
    real_mask = mask.copy()
    real_mask[real_mask == 1] = 255

    # segmentationed image
    masked_img = img * mask

    # 예측 결과 plot
    combined = np.concatenate([img, label, real_mask, masked_img], axis = 1)
    plt.axis('off')
    plt.imshow(combined)
    plt.show()



def plot_model_prediction(model, DEVICE, valid_images, valid_labels):#, save_dir, epochs):
    
    size = (224, 224)
    
    i = random.sample(list(np.arange(0, len(valid_images))), 1)
    i = i[0]
    path_img = valid_images[i]
    path_label = valid_labels[i]
    
    # figure 생성
    fig = plt.gcf()
    fig.set_size_inches(12, 5)
    plt.axis('off')
    
    # eval 전 이미지 전처리
    img = cv2.imread(path_img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    img = cv2.resize(img, size, interpolation = cv2.INTER_LINEAR)
    
    # label 이미지 전처리
    label = cv2.imread(path_label)
    # 양성형 이웃 보간 (2x2 픽셀 참조하여 보간함.)
    label = cv2.resize(label, size, interpolation = cv2.INTER_LINEAR)

    model.eval()
    eval_image = img / 255.0
    eval_image = eval_image.astype(np.float32)
    eval_image = eval_image.transpose((2,0,1))
    eval_image = torch.from_numpy(eval_image).unsqueeze(0) # Batch 채널 추가 -> (1, 3, 256, 256)
    eval_image = eval_image.to( device=DEVICE, dtype = torch.float32 )


    # we do not need to calculate gradients
    with torch.no_grad():
        # Prediction
        pred = model(eval_image)  # (1, 1, 224, 224)
  
    # put on cpu
#     pred = pred.cpu()

    # pass sigloid 
    # pred = torch.sigmoid(pred)
    
#     # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
    
    if isinstance(pred, dict):
        pred = torch.sigmoid(pred['out'])
        
    else:
        pred = torch.sigmoid(pred)  
    
    mask = pred.clone()   # (1, 1, 224, 224)
    
    # 0.5를 기준으로 마스크 만들기.
    mask[mask >= 0.5 ] = 1
    mask[mask < 0.5 ] = 0
    with torch.no_grad(): 
    #   mask = torch.argmax( mask, dim=1 )  # argmax 하기전에는 (1, 2, 224, 224)
      mask = mask.squeeze() # (2, 224, 224) 오류 남 (224, 224)가 되야함!


    mask = mask.to(device = 'cpu', dtype = torch.int64).numpy() # tensor to numpy (반드시 디바이스도 변경)
    mask = np.stack( (mask,)*3, axis=-1 ) # (256,256,3)
    
    # 마스킹을 보여주기 위해 흰색처리
    real_mask = mask.copy()
    real_mask[real_mask == 1] = 255
    
    # segmentationed image
    masked_img = img * mask
    
    # 예측 결과 plot
    combined = np.concatenate([img, label, real_mask, masked_img], axis = 1)
    plt.axis('off')
    plt.imshow(combined)
    plt.show()
    
    
def ploting_incorrect_img( plot_input ):

# color_list = [0, 255,255] # 민트색
# color_list = [255, 155, 0] # 주황형광
  color_list = [252, 0, 255] # 핑크형광
  
  for i in range(len(plot_input)): # i = 0~165

    fig = plt.gcf()
    fig.set_size_inches(15,5)
    plt.axis('off')


    raw_img = plot_input[i][0]    # (224 224 3)
    raw_label = plot_input[i][1]  # (224 224 1)
    labeled_mask = plot_input[i][2]  # (224 224 1)
    inc_labeled_mask = plot_input[i][3] # (224 224 1)
    inc_mask = plot_input[i][4]  # (224 224 1)
    gra_mask = plot_input[i][5] #(244 244 1)

    # raw_label up dim
    raw_label = raw_label.squeeze() # (224 224)
    raw_label = np.stack( (raw_label,)*3, axis=-1 ) # (224 224 3)   0~1

    #labeled_mask up dim
    labeled_mask = labeled_mask.squeeze() # (224 224)
    labeled_mask = np.stack( (labeled_mask,)*3, axis=-1 ) # (224 224 3)  predict  0~1

    #labeled_mask up dim and color change
    inc_labeled_mask = inc_labeled_mask.squeeze() # (224 224) # 0 100 255
    inc_labeled_mask = np.stack( (inc_labeled_mask,)*3, axis=-1 ) # (224 224 3)  predict  0~1

    inc_labeled_mask_0, inc_labeled_mask_1, inc_labeled_mask_2 = inc_labeled_mask[:,:,0], inc_labeled_mask[:,:,1], inc_labeled_mask[:,:,2]
    inc_labeled_mask_0[inc_labeled_mask_0 == 100] = color_list[0]
    inc_labeled_mask_1[inc_labeled_mask_1 == 100] = color_list[1]
    inc_labeled_mask_2[inc_labeled_mask_2 == 100] = color_list[2]
    inc_labeled_mask = inc_labeled_mask/255


    #inc_mask up dim and color change
    inc_mask = inc_mask.squeeze() # (224 224)
    inc_mask = np.stack( (inc_mask,)*3, axis=-1 ) # (224 224 3)    black(0) or incorrect(1)
    # inc_mask[inc_mask == 1][:][:] = [153, 189, 33] # mint color BRG

    # make masked_img
    raw_img_ = raw_img.copy()

    raw_img_0, raw_img_1, raw_img_2 = raw_img_[:,:,0], raw_img_[:,:,1], raw_img_[:,:,2]
    raw_img_0[raw_label[:,:,0] != labeled_mask[:,:,0]] = color_list[0]
    raw_img_1[raw_label[:,:,1] != labeled_mask[:,:,1]] = color_list[1]
    raw_img_2[raw_label[:,:,2] != labeled_mask[:,:,2]] = color_list[2]
    
    
    masked_img = inc_labeled_mask.copy()
    masked_img[masked_img==1] = raw_img_[masked_img==1]

    ##흑백 버전 (흰바)
    #gra_mask = gra_mask.squeeze() # (224 224)
    #gra_mask = np.stack( (gra_mask,)*3, axis=-1 )

    #흑백 버전 (검바)
    gra_mask =1-gra_mask
    gra_mask = gra_mask.squeeze() # (224 224)
    gra_mask = np.stack( (gra_mask,)*3, axis=-1 )

    # #컬러 버전 [255,128, -]
    # gra_mask = gra_mask.squeeze() # (224 224)
    # gra_mask = np.stack( (gra_mask,)*3, axis=-1 )
    # gra_mask[:,:,0]=255/255
    # gra_mask[:,:,1]=155/255
    
    # combined = np.concatenate([raw_img, raw_label, inc_labeled_mask, masked_img,gra_mask], axis = 1)
    combined = np.concatenate([raw_img, raw_label, inc_labeled_mask, gra_mask], axis = 1)
                            #     이미지    마스크   예측마스크  예측마스크(민트)
    
    # print(f'{i}번째 이미지 ')
    plt.imshow(combined)
    plt.show() 

    
def ploting_mode_incorrect_img(plot_input_list):
    
    # color_list = [0, 255,255] # 민트색
    # color_list = [255, 155, 0] # 주황형광
    color_list = [252, 0, 255] # 핑크형광
    
    print('[0:original ,1: None, 2:relu, 3:leakyrelu, 4:tanht, 5:selu]\n1행: image, GT, incorrecting masks_0~5  \n2행: image, GT, gradation masks_0~5')
    # {0:original ,1: None, 2:sigmod, 3:relu, 4:leakyrelu, 5:tanh}
    for i in range(len(plot_input_list[0])): # i = 0~165
        
        fig = plt.gcf()
        fig.set_size_inches(21,10)
        plt.axis('off')  
                
        for j in range(len(plot_input_list)):
            
            globals()['raw_img_{}'.format(j)] = plot_input_list[j][i][0]
            globals()['raw_label_{}'.format(j)] = plot_input_list[j][i][1]
            globals()['labeled_mask_{}'.format(j)] = plot_input_list[j][i][2]
            globals()['inc_labeled_mask_{}'.format(j)] = plot_input_list[j][i][3]
            globals()['inc_mask_{}'.format(j)] = plot_input_list[j][i][4]
            globals()['gra_mask_{}'.format(j)] = plot_input_list[j][i][5]

        raw_img_list=[raw_img_0,raw_img_1,raw_img_2,raw_img_3,raw_img_4,raw_img_5]
        raw_label_list=[raw_label_0,raw_label_1,raw_label_2,raw_label_3,raw_label_4,raw_label_5]
        labeled_mask_list=[labeled_mask_0,labeled_mask_1,labeled_mask_2,labeled_mask_3,labeled_mask_4,labeled_mask_5]
        inc_labeled_mask_list=[inc_labeled_mask_0,inc_labeled_mask_1,inc_labeled_mask_2,inc_labeled_mask_3,inc_labeled_mask_4,inc_labeled_mask_5]
        inc_mask_list=[inc_mask_0,inc_mask_1,inc_mask_2,inc_mask_3,inc_mask_4,inc_mask_5]
        gra_mask_list=[gra_mask_0,gra_mask_1,gra_mask_2,gra_mask_3,gra_mask_4,gra_mask_5]
            
        for k in range(len(plot_input_list)):
            
            # raw_label up dim
            raw_label_list[k] = raw_label_list[k].squeeze() # (224 224)
            raw_label_list[k] = np.stack( (raw_label_list[k],)*3, axis=-1 ) # (224 224 3)   0~1

            #labeled_mask up dim
            labeled_mask_list[k] = labeled_mask_list[k].squeeze() # (224 224)
            labeled_mask_list[k] = np.stack( (labeled_mask_list[k],)*3, axis=-1 ) # (224 224 3)  predict  0~1

            #labeled_mask up dim and color change
            inc_labeled_mask_list[k] = inc_labeled_mask_list[k].squeeze() # (224 224) # 0 100 255
            inc_labeled_mask_list[k] = np.stack( (inc_labeled_mask_list[k],)*3, axis=-1 ) # (224 224 3)  predict  0~1

            inc_labeled_mask__0, inc_labeled_mask__1, inc_labeled_mask__2 = inc_labeled_mask_list[k][:,:,0], inc_labeled_mask_list[k][:,:,1], inc_labeled_mask_list[k][:,:,2]
            inc_labeled_mask__0[inc_labeled_mask__0 == 100] = color_list[0]
            inc_labeled_mask__1[inc_labeled_mask__1 == 100] = color_list[1]
            inc_labeled_mask__2[inc_labeled_mask__2 == 100] = color_list[2]
            inc_labeled_mask_list[k] = inc_labeled_mask_list[k]/255


            #inc_mask up dim and color change
            inc_mask_list[k] = inc_mask_list[k].squeeze() # (224 224)
            inc_mask_list[k] = np.stack( (inc_mask_list[k],)*3, axis=-1 ) # (224 224 3)    black(0) or incorrect(1)
            # inc_mask[inc_mask == 1][:][:] = [153, 189, 33] # mint color BRG

            # make masked_img
            raw_img_ = raw_img_list[k].copy()

            raw_img__0, raw_img__1, raw_img__2 = raw_img_[:,:,0], raw_img_[:,:,1], raw_img_[:,:,2]
            raw_img__0[raw_label_list[k][:,:,0] != labeled_mask_list[k][:,:,0]] = color_list[0]
            raw_img__1[raw_label_list[k][:,:,1] != labeled_mask_list[k][:,:,1]] = color_list[1]
            raw_img__2[raw_label_list[k][:,:,2] != labeled_mask_list[k][:,:,2]] = color_list[2]


            masked_img = inc_labeled_mask_list[k].copy()
            masked_img[masked_img==1] = raw_img_[masked_img==1]
            

            gra_mask_list[k] =1-gra_mask_list[k]
            gra_mask_list[k] = gra_mask_list[k].squeeze() # (224 224)
            gra_mask_list[k] = np.stack( (gra_mask_list[k],)*3, axis=-1 )


        #combined = np.concatenate([raw_img, raw_label, inc_labeled_mask, masked_img, gra_mask], axis = 1)
                                #     이미지    마스크   예측마스크  예측마스크(민트)
        combined_1 = np.concatenate([raw_img_list[0], raw_label_list[0], inc_labeled_mask_list[0],inc_labeled_mask_list[1],inc_labeled_mask_list[2],inc_labeled_mask_list[3],inc_labeled_mask_list[4],inc_labeled_mask_list[5]], axis = 1)  
        combined_2 = np.concatenate([raw_img_list[0], raw_label_list[0], gra_mask_list[0],gra_mask_list[1],gra_mask_list[2],gra_mask_list[3],gra_mask_list[4],gra_mask_list[5]], axis = 1)  
        combined = np.concatenate([combined_1,combined_2], axis = 0)

        print(f'{i}번째 이미지 ')
        plt.imshow(combined)
        plt.show()     
    
    
def result_incorrect_plot(plot_input_list, method=1):
    '''pickle file
    plot_input0 = 그냥 best epoch 피클.load(f)
    plot_input1 = best dice activation 피클.load(f)
    plot_input2 = best dice attention 피클.load(f)
    
    plot_input_list = [plot_input0, plot_input1, plot_input2]
    '''
    
    # color_list = [0, 255,255] # 민트색
    # color_list = [255, 155, 0] # 주황형광
    color_list = [252, 0, 255] # 핑크형광
    
    print('[0:original ,1:best dice activation , 2:best dice attention]\n1행: image, GT, incorrecting masks_0~2  \n2행: image, GT, gradation masks_0~2')
    for i in range(len(plot_input_list[0])): # i = 0~165
        
        fig = plt.gcf()
        fig.set_size_inches(18,8)
        plt.axis('off')  
                
        for j in range(len(plot_input_list)):  # len(plot_input_list) : 3 /  j : (0,1,2)
            
            globals()['raw_img_{}'.format(j)] = plot_input_list[j][i][0]
            globals()['raw_label_{}'.format(j)] = plot_input_list[j][i][1]
            globals()['labeled_mask_{}'.format(j)] = plot_input_list[j][i][2]
            globals()['inc_labeled_mask_{}'.format(j)] = plot_input_list[j][i][3]
            globals()['inc_mask_{}'.format(j)] = plot_input_list[j][i][4]
            globals()['gra_mask_{}'.format(j)] = plot_input_list[j][i][5]

        raw_img_list=[raw_img_0,raw_img_1,raw_img_2]
        raw_label_list=[raw_label_0,raw_label_1,raw_label_2]
        labeled_mask_list=[labeled_mask_0,labeled_mask_1,labeled_mask_2]
        inc_labeled_mask_list=[inc_labeled_mask_0,inc_labeled_mask_1,inc_labeled_mask_2]
        inc_mask_list=[inc_mask_0,inc_mask_1,inc_mask_2]
        gra_mask_list=[gra_mask_0,gra_mask_1,gra_mask_2]
            
        for k in range(len(plot_input_list)): # len(plot_input_list): 3  /  k = (0,1,2)
            
            # raw_label up dim
            raw_label_list[k] = raw_label_list[k].squeeze() # (224 224)
            raw_label_list[k] = np.stack( (raw_label_list[k],)*3, axis=-1 ) # (224 224 3)   0~1

            #labeled_mask up dim
            labeled_mask_list[k] = labeled_mask_list[k].squeeze() # (224 224)
            labeled_mask_list[k] = np.stack( (labeled_mask_list[k],)*3, axis=-1 ) # (224 224 3)  predict  0~1

            #labeled_mask up dim and color change
            inc_labeled_mask_list[k] = inc_labeled_mask_list[k].squeeze() # (224 224) # 0 100 255
            inc_labeled_mask_list[k] = np.stack( (inc_labeled_mask_list[k],)*3, axis=-1 ) # (224 224 3)  predict  0~1

            inc_labeled_mask__0, inc_labeled_mask__1, inc_labeled_mask__2 = inc_labeled_mask_list[k][:,:,0], inc_labeled_mask_list[k][:,:,1], inc_labeled_mask_list[k][:,:,2]
            inc_labeled_mask__0[inc_labeled_mask__0 == 100] = color_list[0]
            inc_labeled_mask__1[inc_labeled_mask__1 == 100] = color_list[1]
            inc_labeled_mask__2[inc_labeled_mask__2 == 100] = color_list[2]
            inc_labeled_mask_list[k] = inc_labeled_mask_list[k]/255


            #inc_mask up dim and color change
            inc_mask_list[k] = inc_mask_list[k].squeeze() # (224 224)
            inc_mask_list[k] = np.stack( (inc_mask_list[k],)*3, axis=-1 ) # (224 224 3)    black(0) or incorrect(1)
            # inc_mask[inc_mask == 1][:][:] = [153, 189, 33] # mint color BRG

            # make masked_img
            raw_img_ = raw_img_list[k].copy()

            raw_img__0, raw_img__1, raw_img__2 = raw_img_[:,:,0], raw_img_[:,:,1], raw_img_[:,:,2]
            raw_img__0[raw_label_list[k][:,:,0] != labeled_mask_list[k][:,:,0]] = color_list[0]
            raw_img__1[raw_label_list[k][:,:,1] != labeled_mask_list[k][:,:,1]] = color_list[1]
            raw_img__2[raw_label_list[k][:,:,2] != labeled_mask_list[k][:,:,2]] = color_list[2]


            masked_img = inc_labeled_mask_list[k].copy()
            masked_img[masked_img==1] = raw_img_[masked_img==1]
            

            gra_mask_list[k] =1-gra_mask_list[k]
            gra_mask_list[k] = gra_mask_list[k].squeeze() # (224 224)
            gra_mask_list[k] = np.stack( (gra_mask_list[k],)*3, axis=-1 )

        if method == 1:
            
            combined_1 = np.concatenate([raw_img_list[0], raw_label_list[0], inc_labeled_mask_list[0],inc_labeled_mask_list[1],inc_labeled_mask_list[2]], axis = 1)  
                                        #   원본 이미지     정답 마스크            1장 마스크        2장 best dice act 마스크    2장 best dice att 마스크

            combined_2 = np.concatenate([np.clip(raw_img_list[0]+255,255,255) , np.clip(raw_label_list[0]+255,255,255), # 원본 이미지, 라벨 전부 흰색 만들기
                                         gra_mask_list[0],gra_mask_list[1],gra_mask_list[2]], axis = 1)

            combined = np.concatenate([combined_1,combined_2], axis = 0)
            
        elif method == 2:
            combined_1 = np.concatenate([raw_img_list[0], inc_labeled_mask_list[0],inc_labeled_mask_list[1],inc_labeled_mask_list[2]], axis = 1)  
                                    

            combined_2 = np.concatenate([raw_label_list[0], gra_mask_list[0], gra_mask_list[1], gra_mask_list[2]], axis = 1)

            combined = np.concatenate([combined_1,combined_2], axis = 0)  
            
        elif method == 3:
            combined_1 = np.concatenate([raw_img_list[0],inc_labeled_mask_list[0],inc_labeled_mask_list[1]], axis = 1)  
                                    

            combined_2 = np.concatenate([raw_label_list[0],gra_mask_list[0], gra_mask_list[1]], axis = 1)

            combined = np.concatenate([combined_1,combined_2], axis = 0)   

        print(f'{i}번째 이미지 ')
        plt.imshow(combined)
        plt.show() 

def result(plot_input_list):
    '''pickle file
    plot_input0 = 그냥 best epoch 피클.load(f)
    plot_input1 = best dice activation 피클.load(f)
    plot_input2 = best dice attention 피클.load(f)
    
    plot_input_list = [plot_input0, plot_input1, plot_input2]
    '''
    
    # color_list = [0, 255,255] # 민트색
    # color_list = [255, 155, 0] # 주황형광
    color_list = [252, 0, 255] # 핑크형광
    
    print('[0:original ,1:best dice activation , 2:best dice attention]\n1행: image, GT, incorrecting masks_0~2  \n2행: image, GT, gradation masks_0~2')

    for i in range(len(plot_input_list[0])): # i = 0~165
        
        fig = plt.gcf()
        fig.set_size_inches(18,8)
        plt.axis('off')  
                
        for j in range(len(plot_input_list)):  # len(plot_input_list) : 4 /  j : (0,1,2,3)
            
            globals()['raw_img_{}'.format(j)] = plot_input_list[j][i][0]
            globals()['raw_label_{}'.format(j)] = plot_input_list[j][i][1]
            globals()['labeled_mask_{}'.format(j)] = plot_input_list[j][i][2]
            globals()['inc_labeled_mask_{}'.format(j)] = plot_input_list[j][i][3]
            globals()['inc_mask_{}'.format(j)] = plot_input_list[j][i][4]
            globals()['gra_mask_{}'.format(j)] = plot_input_list[j][i][5]

        raw_img_list=[raw_img_0,raw_img_1,raw_img_2, raw_img_3]
        raw_label_list=[raw_label_0,raw_label_1,raw_label_2,raw_label_3]
        labeled_mask_list=[labeled_mask_0,labeled_mask_1,labeled_mask_2,labeled_mask_3]
        inc_labeled_mask_list=[inc_labeled_mask_0,inc_labeled_mask_1,inc_labeled_mask_2,inc_labeled_mask_3]
        inc_mask_list=[inc_mask_0,inc_mask_1,inc_mask_2,inc_mask_3]
        gra_mask_list=[gra_mask_0,gra_mask_1,gra_mask_2,gra_mask_3]
            
        for k in range(len(plot_input_list)): # len(plot_input_list): 3  /  k = (0,1,2)
            
            # raw_label up dim
            raw_label_list[k] = raw_label_list[k].squeeze() # (224 224)
            raw_label_list[k] = np.stack( (raw_label_list[k],)*3, axis=-1 ) # (224 224 3)   0~1

            #labeled_mask up dim
            labeled_mask_list[k] = labeled_mask_list[k].squeeze() # (224 224)
            labeled_mask_list[k] = np.stack( (labeled_mask_list[k],)*3, axis=-1 ) # (224 224 3)  predict  0~1

            #labeled_mask up dim and color change
            inc_labeled_mask_list[k] = inc_labeled_mask_list[k].squeeze() # (224 224) # 0 100 255
            inc_labeled_mask_list[k] = np.stack( (inc_labeled_mask_list[k],)*3, axis=-1 ) # (224 224 3)  predict  0~1

            inc_labeled_mask__0, inc_labeled_mask__1, inc_labeled_mask__２ = inc_labeled_mask_list[k][:,:,0], inc_labeled_mask_list[k][:,:,1], inc_labeled_mask_list[k][:,:,2]
            inc_labeled_mask__0[inc_labeled_mask__0 == 100] = color_list[0]
            inc_labeled_mask__1[inc_labeled_mask__1 == 100] = color_list[1]
            inc_labeled_mask__2[inc_labeled_mask__2 == 100] = color_list[2]
            inc_labeled_mask_list[k] = inc_labeled_mask_list[k]/255


            #inc_mask up dim and color change
            inc_mask_list[k] = inc_mask_list[k].squeeze() # (224 224)
            inc_mask_list[k] = np.stack( (inc_mask_list[k],)*3, axis=-1 ) # (224 224 3)    black(0) or incorrect(1)
            # inc_mask[inc_mask == 1][:][:] = [153, 189, 33] # mint color BRG

            # make masked_img
            raw_img_ = raw_img_list[k].copy()

            raw_img__0, raw_img__1, raw_img__2 = raw_img_[:,:,0], raw_img_[:,:,1], raw_img_[:,:,2]
            raw_img__0[raw_label_list[k][:,:,0] != labeled_mask_list[k][:,:,0]] = color_list[0]
            raw_img__1[raw_label_list[k][:,:,1] != labeled_mask_list[k][:,:,1]] = color_list[1]
            raw_img__2[raw_label_list[k][:,:,2] != labeled_mask_list[k][:,:,2]] = color_list[2]


            masked_img = inc_labeled_mask_list[k].copy()
            masked_img[masked_img==1] = raw_img_[masked_img==1]
            

            gra_mask_list[k] =1-gra_mask_list[k]
            gra_mask_list[k] = gra_mask_list[k].squeeze() # (224 224)
            gra_mask_list[k] = np.stack( (gra_mask_list[k],)*3, axis=-1 )


        combined_1 = np.concatenate([raw_img_list[0], inc_labeled_mask_list[0],inc_labeled_mask_list[1],inc_labeled_mask_list[2],inc_labeled_mask_list[3]], axis = 1)  

        combined_2 = np.concatenate([raw_label_list[0], gra_mask_list[0], gra_mask_list[1], gra_mask_list[2],gra_mask_list[3]], axis = 1)

        combined = np.concatenate([combined_1,combined_2], axis = 0)            

        print(f'{i}번째 이미지 ')
        plt.imshow(combined)
        plt.show()     


def get_interval(center, interval_width):
    # 중심지점에서 interval_width/2 만큼 떨어진 시작점과 끝점 계산
    start_point = center - interval_width/2
    end_point = center + interval_width/2
   
    return start_point, end_point



def find_point(data,center=0.5,percent=0.05,step=0.001):
    
    prob_series = pd.Series(data)
    area=0
    i=0
    
    while area <= percent:
        i+=1
        start_point, end_point=get_interval(center,step*i)
    
        cut_prob=prob_series[prob_series>start_point][prob_series<end_point]
        area=len(cut_prob)/len(prob_series)
    print("start_point: ",start_point,"\nend_point: ",end_point,'\narea:',area,'(',len(cut_prob),'/',len(prob_series),')')
    return start_point,end_point,area

def distribution_plot_n_line(prob_dict,center=0.5,percent=0.05,step=0.001):
    fig = plt.figure(figsize=(14,4))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    axes = [ax1,ax2,ax3]
    keys = ['all', 'correct_all', 'incorrect_all']
    
    for i,(ax, key) in enumerate(zip(axes,keys)):
        print('\n',key,' :\n')
        start_point,end_point,__=find_point(prob_dict[key][0],center=0.5,percent=0.05,step=0.001)
        _=sns.distplot(prob_dict[key][0], ax=ax)
        _=ax.set_xlabel("prob")
        _=ax.set_ylabel("density")
        _=ax.set_title(key,fontsize = 16)
        _=ax.axvline(x=start_point, color='r', linestyle='--')
        _=ax.text(x=start_point-0.2,y=round(max([h.get_height() for h in ax.patches])/2,0),s=round(start_point,3))
        _=ax.axvline(x=end_point, color='r', linestyle='--')
        _=ax.text(x=end_point+0.01,y=round(max([h.get_height() for h in ax.patches])/2,0),s=round(end_point,3))

    plt.show()
    
    
def distribution_plot_modes(prob_dict_list,center=0.5,percent=0.05,step=0.001):
    fig = plt.figure(figsize=(16,12))
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)
    axes = [ax1,ax2,ax3, ax4, ax5 ,ax6]
    titles = ['Sigmoid' ,'None+Att', 'ReLU+Att', 'LeakyRelU+Att', 'Tanht+Att', 'SELU+Att']
    
    for i,(ax, title) in enumerate(zip(axes,titles)):
        print('\n',title,' :\n')
        start_point,end_point=find_point(prob_dict_list[i]['incorrect_all'][0],center=center,percent=percent,step=step)
        _=sns.distplot(prob_dict_list[i]['incorrect_all'][0], ax=ax)
        _=ax.set_xlabel("prob")
        _=ax.set_ylabel("density")
        _=ax.set_title(title,fontsize = 16)
        _=ax.axvline(x=start_point, color='r', linestyle='--')
        _=ax.text(x=start_point-0.15,y=round(max([h.get_height() for h in ax.patches])/2,0),s=round(start_point,3))
        _=ax.axvline(x=end_point, color='r', linestyle='--')
        _=ax.text(x=end_point+0.01,y=round(max([h.get_height() for h in ax.patches])/2,0),s=round(end_point,3))

    plt.show()
    
def result_distribution_plot(prob_dict_list,center=0.5,percent=0.05,step=0.001):
    fig = plt.figure(figsize=(20,5))
    ax1 = fig.subplots(1,3)

    axes = [ax1[0], ax1[1], ax1[2]]
    titles = ['Sigmoid' ,'Best Act', 'Best CBAM']
    
    for i,(ax, title) in enumerate(zip(axes,titles)):
        print('\n',title,' :\n')
        start_point,end_point, area=find_point(prob_dict_list[i]['incorrect_all'][0],center=center,percent=percent,step=step)
        _=sns.distplot(prob_dict_list[i]['incorrect_all'][0], ax=ax)
        _=ax.set_xlabel("prob")
        _=ax.set_ylabel("density")
        _=ax.set_title(title,fontsize = 16)
        _=ax.axvline(x=start_point, color='r', linestyle='--')
        _=ax.text(x=start_point-0.15,y=round(max([h.get_height() for h in ax.patches])/2,0),s=round(start_point,3))
        _=ax.axvline(x=end_point, color='r', linestyle='--')
        _=ax.text(x=end_point+0.01,y=round(max([h.get_height() for h in ax.patches])/2,0),s=round(end_point,3))

    plt.show()
    
def distribution_plot(prob_dict):
    fig = plt.figure(figsize=(32,8))
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)
    axes = [ax1,ax2,ax3]
    keys = ['all', 'correct_all', 'incorrect_all']

    for i,(ax, key) in enumerate(zip(axes,keys)):
        _=sns.distplot(prob_dict[key][0], ax=ax)
        _=ax.set_xlabel("prob", fontsize=30)
        _=ax.set_ylabel("density", fontsize=30)
        _=ax.tick_params(axis='x', labelsize=25)
        _=ax.tick_params(axis='y', labelsize=20)
        _=ax.set_title(key,fontsize = 40)
        #_=ax.set_ylim([0, 5])
    #plt.tight_layout()
    plt.show()
    
    
    
# def distribution_plot(prob_dict):
#     fig = plt.figure(figsize=(32,8))
#     ax1 = fig.add_subplot(1, 3, 1)
#     ax2 = fig.add_subplot(1, 3, 2)
#     ax3 = fig.add_subplot(1, 3, 3)
#     axes = [ax1,ax2,ax3]
#     keys = ['all', 'correct_all', 'incorrect_all']

#     for i,(ax, key) in enumerate(zip(axes,keys)):
#         _=sns.distplot(prob_dict[key][0], ax=ax)
#         _=ax.set_xlabel("prob", fontsize=30)
#         _=ax.set_ylabel("density", fontsize=30)
#         _=ax.tick_params(axis='x', labelsize=25)
#         _=ax.tick_params(axis='y', labelsize=20)
#         _=ax.set_title(key,fontsize = 40)
#     plt.tight_layout()
#     plt.show()
    
    
    
    
    
def scatter_loss(beta, lamb, mode, m=0.5):
    
    weights_path = '/home/sh/lab/YaML/Ot-Unet_best_94_0.952.pth'
    weights = torch.load(weights_path)
    model=Unet().to(device)
    model.load_state_dict(weights['net'])

    prob_dict , plot_input = extract_inference_prob_n_label('test',model,8,'/home/sh/lab/YaML/data/Ottawa-Dataset/') 

    output=prob_dict[mode][0]
    # 예측값이 0이거나 1인 경우 log 함수의 값이 계산되지 않으므로 작은 값을 더해줌
    output=np.clip(output, 1e-15, 1 - 1e-15)
    
    def tanh(x):
        return (1+np.tanh(x))*(1-np.tanh(x))


    loss1 = beta * tanh( lamb * ( np.array(output) - m ) )
    print('CustomLoss Mean:', np.mean(loss1))

    #test셋 라벨들 flatten해서 리스트 하나에 넣기 / plot_dict['all']


    """
    label_f: 실제 레이블(0 또는 1)
    output: 모델이 예측한 확률값
    """

    label_f=prob_dict[mode][1]

    # BCE Loss 계산
    loss2 = -( label_f * np.log(output) + (1 - np.array(label_f)) * np.log(1-np.array(output)))
    print('BasicLoss Mean:',np.mean(loss2))


    loss3=loss1*loss2
    print('CL*BL Mean:',np.mean(loss3))


    loss4 = loss1 + loss2
    print('CL+BL Mean:',np.mean(loss4))


    fig = plt.figure(figsize=(14,10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    titles = ['CustomLoss', 'BCE Loss', 'customLoss * BCE Loss', 'customLoss + BCE Loss']
    losses = [loss1,loss2,loss3,loss4]

    data = pd.DataFrame({'output': output, 'loss1': loss1, 'loss2': loss2, 'loss3': loss3,'loss4': loss4})
    _=sns.scatterplot(x='output', y='loss1', data=data,ax=ax1)
    _=ax1.set_title(titles[0],fontsize = 16)
    _=sns.scatterplot(x='output', y='loss2', data=data,ax=ax2)
    _=ax2.set_title(titles[1],fontsize = 16)
    _=sns.scatterplot(x='output', y='loss3', data=data,ax=ax3)
    _=ax3.set_title(titles[2],fontsize = 16)
    _=sns.scatterplot(x='output', y='loss4', data=data,ax=ax4)
    _=ax4.set_title(titles[3],fontsize = 16)

    _=plt.tight_layout()
    
    
def scatter_loss_sig(beta, lamb, mode, m=0.5):
    
    weights_path = '/home/sh/lab/YaML/Ot-Unet_best_94_0.952.pth'
    weights = torch.load(weights_path)
    model=Unet().to(device)
    model.load_state_dict(weights['net'])

    prob_dict , plot_input = extract_inference_prob_n_label('test',model,8,'/home/sh/lab/YaML/data/Ottawa-Dataset/') 

    output=prob_dict[mode][0]
    # 예측값이 0이거나 1인 경우 log 함수의 값이 계산되지 않으므로 작은 값을 더해줌
    output=np.clip(output, 1e-15, 1 - 1e-15)
    
    def sigmoid(x, deff=False):
        if deff:
            return sigmoid(x)*(1-sigmoid(x))
        else:
            return 1 / (1 + math.exp(-x))
    
    def sig_diff(x):

        return sigmoid(x)*(1-sigmoid(x))


    loss1 = beta * sig_diff( lamb * ( np.array(output) - m ) )
    print('CustomLoss Mean:', np.mean(loss1))

    #test셋 라벨들 flatten해서 리스트 하나에 넣기 / plot_dict['all']


    """
    label_f: 실제 레이블(0 또는 1)
    output: 모델이 예측한 확률값
    """

    label_f=prob_dict[mode][1]

    # BCE Loss 계산
    loss2 = -( label_f * np.log(output) + (1 - np.array(label_f)) * np.log(1-np.array(output)))
    print('BasicLoss Mean:',np.mean(loss2))


    loss3=loss1*loss2
    print('CL*BL Mean:',np.mean(loss3))


    loss4 = loss1 + loss2
    print('CL+BL Mean:',np.mean(loss4))


    fig = plt.figure(figsize=(14,10))
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)
    titles = ['CustomLoss', 'BCE Loss', 'customLoss * BCE Loss', 'customLoss + BCE Loss']
    losses = [loss1,loss2,loss3,loss4]

    data = pd.DataFrame({'output': output, 'loss1': loss1, 'loss2': loss2, 'loss3': loss3,'loss4': loss4})
    _=sns.scatterplot(x='output', y='loss1', data=data,ax=ax1)
    _=ax1.set_title(titles[0],fontsize = 16)
    _=sns.scatterplot(x='output', y='loss2', data=data,ax=ax2)
    _=ax2.set_title(titles[1],fontsize = 16)
    _=sns.scatterplot(x='output', y='loss3', data=data,ax=ax3)
    _=ax3.set_title(titles[2],fontsize = 16)
    _=sns.scatterplot(x='output', y='loss4', data=data,ax=ax4)
    _=ax4.set_title(titles[3],fontsize = 16)

    _=plt.tight_layout()
    
    
def score_plot(epoches,state,train_dice_lst,train_iou_lst,train_loss_lst,valid_dice_lst,valid_iou_lst,valid_loss_lst,start_epoch=1):
    fig = plt.figure(figsize=(20,20))
    axs = fig.subplots(3,1)

    train_score = {'Dice':train_dice_lst, 'Iou':train_iou_lst, 'Loss':train_loss_lst}
    valid_score = {'Dice':valid_dice_lst, 'Iou':valid_iou_lst, 'Loss':valid_loss_lst}

    for i in range(len(train_score)):
        axs[i].set_title(list(train_score.keys())[i], fontsize=30 ,fontweight='bold')
        axs[i].set_xlabel('Epoch', fontsize=20, fontweight='bold')
        axs[i].set_ylabel(list(train_score.keys())[i], fontsize=20, fontweight='bold')
        axs[i].plot(range(start_epoch,epoches+1),train_score[list(train_score.keys())[i]], label='Train')
        axs[i].plot(range(start_epoch,epoches+1),valid_score[list(valid_score.keys())[i]], label='Valid')
        if i == 2:
            axs[i].legend(loc='upper right', fontsize=30)
        else:
            axs[i].legend(loc='upper left', fontsize=30)

        axs[i].axvline(x=state['epoch'], color='r', linestyle='--')


    plt.tight_layout()
    plt.show()
    
    
    


def find_point_select(data,start_point,end_point):
    
    prob_series = pd.Series(data)
    cut_prob=prob_series[prob_series>start_point][prob_series<end_point]
    area=len(cut_prob)/len(prob_series)
    print("start_point: ",start_point,"\nend_point: ",end_point,'\narea:',area,'(',len(cut_prob),'/',len(prob_series),')')
    return area
    

    
def distribution_plot_n_line_incorrect_gab(input1,input2,center=0.5,percent=0.05,step=0.001):
    fig = plt.figure(figsize=(8,14))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)
    axes = [ax1,ax2]
    keys = ['A model', 'B model']
    
    print(f"{keys[0]}'s information ")
    start_point1,end_point1,area1=find_point(input1['incorrect_all'][0],center=center,percent=percent,step=step)
    
    print(f"{keys[1]}'s information ")
    start_point2,end_point2,area2=find_point(input2['incorrect_all'][0],center=center,percent=percent,step=step)
    
    _=sns.distplot(input1['incorrect_all'][0], ax=ax1)
    _=ax1.set_xlabel("prob",fontsize=15)
    _=ax1.set_ylabel("density",fontsize=15)
    _=ax1.set_title(keys[0],fontsize = 20)
    _=ax1.axvline(x=start_point1, color='k',linestyle='--')
    #_=ax1.text(x=start_point1-0.12,y=round(max([h.get_height() for h in ax1.patches])/2,0),s=round(start_point1,3), fontsize=15)
    _=ax1.axvline(x=end_point1, color='k',linestyle='--')
    _=ax1.tick_params(axis='x', labelsize=15)
   # _=ax1.text(x=end_point1+0.01,y=round(max([h.get_height() for h in ax1.patches])/2,0),s=round(end_point1,3), fontsize=15)
    
    __=sns.distplot(input2['incorrect_all'][0], ax=ax2)
    __=ax2.set_xlabel("prob",fontsize=15)
    __=ax2.set_ylabel("density",fontsize=15)
    __=ax2.set_title(keys[1],fontsize = 20)
    __=ax2.axvline(x=start_point1, color='k',linestyle='--')
    #__=ax2.text(x=start_point1-0.14,y=round(max([h.get_height() for h in ax2.patches])*3/4,0),s=round(start_point1,3),color='r', fontsize=20)
    __=ax2.axvline(x=end_point1, color='k',linestyle='--')
    #__=ax2.text(x=end_point1+0.03,y=round(max([h.get_height() for h in ax2.patches])*3/4,0),s=round(end_point1,3),color='r', fontsize=20)
    
    __=ax2.axvline(x=start_point2, color='r', linestyle='--')
    #__=ax2.text(x=start_point1-0.13,y=round(max([h.get_height() for h in ax2.patches])/2,0),s=round(start_point2,3), fontsize=20)
    __=ax2.axvline(x=end_point2, color='r', linestyle='--')
    #__=ax2.text(x=end_point2+0.01,y=round(max([h.get_height() for h in ax2.patches])/2,0),s=round(end_point2,3), fontsize=20)
    __=ax2.tick_params(axis='x', labelsize=15)
    
    
    print(f'{keys[0]}의 라인으로 만들어진 구역의 {keys[1]} information ')
    area1_2=find_point_select(input2['incorrect_all'][0],start_point1,end_point1)
    print(area1_2/area1)
    

    plt.show()
