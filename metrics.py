import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# to be used for validation during training
def iou_pytorch_eval(outputs: torch.Tensor, labels: torch.Tensor):

    # comment out if your model contains a sigmoid or equivalent activation layer
    # outputs = torch.sigmoid(outputs)
    
    # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
    if isinstance(outputs, dict):
        outputs = torch.sigmoid(outputs['out'])
    else:
        outputs = torch.sigmoid(outputs)          

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # (BATCH, 1, H, W) -> (BATCH, H, W)
    labels = labels.squeeze(1).byte()


    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.mean()

# -----------------------------------------------------------------------------
# to be used for validation/testing of trainied models

def mean_iou_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
    # intersection = tp
    # union = tp + fp + fn
    # iou = tp / (tp + fp + fn) = intersection / union

    # BATCH x H x W, need because we process images sequentially in a for-loop
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)
    SMOOTH = 1e-8


    # thresholding since that's how we will make predictions on new imputs (class 0)
    outputs0 = outputs < 0.5
    labels0 = labels < 0.5
    intersection = (outputs0 & labels0).float().sum((1, 2))  # Will be zero if Truth=1 or Prediction=1
    union = (outputs0 | labels0).float().sum((1, 2))         # Will be zero if both are 1
    iou0 = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    # thresholding since that's how we will make predictions on new imputs (class 1)
    outputs1 = outputs > 0.5
    labels1 = labels > 0.5
    intersection = (outputs1 & labels1).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs1 | labels1).float().sum((1, 2))         # Will be zero if both are 0
    iou1 = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return ((iou0 + iou1) / 2).mean()


def iou_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
    # intersection = tp
    # union = tp + fp + fn
    # iou = tp / (tp + fp + fn) = intersection / union

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.mean()

def dice_pytorch_eval(outputs: torch.Tensor, labels: torch.Tensor):

    # comment out if your model contains a sigmoid or equivalent activation layer
    # outputs = torch.sigmoid(outputs)
    
    # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
    if isinstance(outputs, dict):
        outputs = torch.sigmoid(outputs['out'])
    else:
        outputs = torch.sigmoid(outputs)  
    
    
    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5

    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1).byte()  # (BATCH, 1, H, W) -> (BATCH, H, W)
    labels = labels.squeeze(1).byte()


    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    dice = 2 * (intersection + SMOOTH) / (intersection + union + SMOOTH) # We smooth our devision to avoid 0/0

    return dice.mean()
# -----------------------------------------------------------------------------

def dice_score(outputs: torch.Tensor, labels: torch.Tensor):

    # comment out if your model contains a sigmoid or equivalent activation layer
    # outputs = torch.sigmoid(outputs)
    
    # dict형태로 데이터가 들어오는 경우가 있음 
    if isinstance(outputs, dict):
        outputs = outputs['out']

    outputs = F.softmax(outputs, dim=1)
    outputs = torch.argmax(outputs, dim=1)
    
    labels_1=labels[:,1,:,:]

    outputs = outputs.byte()
    labels_1 = labels_1.contiguous().byte()

    SMOOTH = 1e-8
    intersection = (outputs & labels_1).float().sum((1,2)) # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels_1).float().sum((1,2))         # Will be zero if both are 0
    dice = 2 * (intersection + SMOOTH) / (intersection + union + SMOOTH) # We smooth our devision to avoid 0/0

    return dice.mean()

def iou(outputs: torch.Tensor, labels: torch.Tensor):

    # comment out if your model contains a sigmoid or equivalent activation layer
    # outputs = torch.sigmoid(outputs)
    
    # dict형태로 데이터가 들어오는 경우가 있음 #####
    if isinstance(outputs, dict):
        outputs = outputs['out'] 

    outputs = F.softmax(outputs, dim=1) #(8,2,224,224)
    outputs = torch.argmax(outputs, dim=1) #(8,224,224)
    
    labels_1=labels[:,1,:,:]

    outputs = outputs.byte()
    labels_1 = labels_1.contiguous().byte()

    SMOOTH = 1e-8
    intersection = (outputs & labels_1).float().sum((1,2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels_1).float().sum((1,2))         # Will be zero if both are 0
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0

    return iou.mean()



def dice_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
    # intersection = tp
    # union = tp + fp + fn
    # dice = 2 * tp / (2 * tp + fp + fn) = 2*intersection / (intersection + union)

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zero if both are 0
    dice = (2*intersection + SMOOTH) / (intersection + union + SMOOTH)  # We smooth our devision to avoid 0/0

    return dice.mean()


def precision_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
    # intersection = tp
    # tpfp = tp + fp
    # precision = tp / (tp + fp) = intersection / tpfp

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    tpfp = (labels).float().sum((1, 2))                    # Will be zero if both are 0
    precision = (intersection + SMOOTH) / (tpfp + SMOOTH)  # We smooth our devision to avoid 0/0

    return precision.mean()


def recall_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):
    # intersection = tp
    # tpfn = tp + fn
    # recall = tp / (tp + fn) = intersection / tpfn


    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    tpfn = (outputs).float().sum((1, 2))                   # Will be zero if both are 0
    recall = (intersection + SMOOTH) / (tpfn + SMOOTH)     # We smooth our devision to avoid 0/0

    return recall.mean()


def fbeta_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor, beta:float):
    # intersection = tp
    #
    # tpfp = tp + fp
    # precision = tp / (tp + fp) = intersection / tpfp
    #
    # tpfn = tp + fn
    # recall = tp / (tp + fn) = intersection / tpfn
    #
    # fbeta = (1 + beta^2) * (precision * recall) / ((beta^2 * precision) + recall)
    # https://www.quora.com/What-is-the-F2-score-in-machine-learning

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    SMOOTH = 1e-8
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0

    tpfn = (outputs).float().sum((1, 2))                   # Will be zero if both are 0
    recall = (intersection + SMOOTH) / (tpfn + SMOOTH)     # We smooth our devision to avoid 0/0

    tpfp = (labels).float().sum((1, 2))                    # Will be zero if both are 0
    precision = (intersection + SMOOTH) / (tpfp + SMOOTH)  # We smooth our devision to avoid 0/0

    f_beta = (1 + beta ** 2) * (precision * recall) / ((beta **2 * precision) + recall)

    return f_beta.mean()


def accuracy_pytorch_test(outputs: torch.Tensor, labels: torch.Tensor):

    # BATCH x H x W
    assert len(outputs.shape) == 3
    assert len(labels.shape) == 3

    # comment out if your model contains a sigmoid or equivalent activation layer
    outputs = torch.sigmoid(outputs)

    # thresholding since that's how we will make predictions on new imputs
    outputs = outputs > 0.5
    labels = labels > 0.5

    acc = (outputs == labels).float().mean((1, 2))

    return acc.mean()



# -----------------------------------------------------------------------------
# Credit to: https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch


# TODO: replace all `size_average` arguments with `reduction`, see PyTorch docs
#       for any of the loss functions

class IoULoss(nn.Module):
    def __init__(self):
        super(IoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = (intersection + smooth)/(union + smooth)

        return -IoU


class IoUBCELoss(nn.Module):
    def __init__(self):
        super(IoUBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)

        #intersection is equivalent to True Positive count
        #union is the mutually inclusive area of all labels & predictions
        intersection = (inputs * targets).sum()
        total = (inputs + targets).sum()
        union = total - intersection

        IoU = - (intersection + smooth)/(union + smooth)

        # https://pytorch.org/docs/stable/nn.functional.html#binary-cross-entropy
        BCE = F.binary_cross_entropy(input=inputs, target=targets, reduction='mean')
        IoU_BCE = BCE + IoU

        return IoU_BCE

    
#class DiceBCELoss(nn.Module):
#    def __init__(self, weight=None, size_average=True):
#        super(DiceBCELoss, self).__init__()
#
#    def forward(self, inputs, targets, smooth=1):
#        
#        #comment out if your model contains a sigmoid or equivalent activation layer
#        # inputs = torch.sigmoid(inputs)  
#            
#        
#        #flatten label and prediction tensors
#        inputs_ = inputs.clone()
#        targets_ = targets.clone()
#        
#        inputs_ = inputs_[:,1,:,:]
#        inputs_ = inputs_.unsqueeze(1)
#        inputs_ = inputs_.contiguous().view(-1)
#        
#        inputs = inputs.view(-1)
#        
#        targets = torch.stack((targets , targets)).view(-1)
#        targets_ = targets_.view(-1)
        

#        # targets_ = targets.view(-1)
        
#        intersection = (inputs_ * targets_).sum()                            
#        dice_loss = 1 - (2.*intersection + smooth)/(inputs_.sum() + targets_.sum() + smooth)  
#        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
#        Dice_BCE = BCE + dice_loss
        
#        return Dice_BCE    
    
class DiceCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
        if isinstance(inputs, dict):
            inputs = inputs['out']
        
        CE = F.cross_entropy(inputs, targets, reduction='mean')
        
        inputs = F.softmax(inputs, dim=1)

        inputs_1=inputs[:,1,:,:].unsqueeze(dim=1)
        targets_1=targets[:,1,:,:].unsqueeze(dim=1)


        # flatten label and prediction images, leave BATCH and NUM_CLASSES
        # (BATCH, NUM_CLASSES, H, W) -> (BATCH, NUM_CLASSES, H * W)
        inputs_1 = inputs_1.contiguous().view(-1)
        targets_1 = targets_1.contiguous().view(-1)

        # get one number per each 2D image/mask pair
        # .sum(2) : (BATCH, NUM_CLASSES, H * W) -> (BATCH, NUM_CLASSES)
        #dice score을 위해 구할떄는 이진화 시켜서 구하는 반면 loss를 위해 값을 구할때는 이진화 없이 확률값을 이용해서 진행한다
        
        intersection = (inputs_1 * targets_1).sum()
        dice_coef = (2.*intersection + smooth)/(inputs_1.sum() + targets_1.sum() + smooth)
        dice_loss = 1 - dice_coef

        Dice_CE = CE + dice_loss        
        return Dice_CE  

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        # inputs = torch.sigmoid(inputs)  
        
        # dict형태로 데이터가 들어오는 경우가 있음 ######################################################################
        if isinstance(inputs, dict):
            inputs = torch.sigmoid(inputs['out'])
        else:
            inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + dice_loss
        
        return Dice_BCE    
# -----------------------------------------------------------------------------

class mIoULossBinary(nn.Module):
    def __init__(self, weight=None):
        super(mIoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # (BATCH, 1, H, W)
        # we care about both classes represented as 0 and 1 on one the masks

        if self.weight is not None:
            assert self.weight.shape == (targets.shape[1], )
        # make a copy not to change the default weight in the instance of DiceLossMulticlass
        weight = self.weight.copy()

        # invert what is target and what is not
        # 0 -> (-1) * (0 - 1) = 1
        # 1 -> (-1) * (1 - 1) = 0
        targets_inv = (-1) * (targets - 1)

        if weight is None:
            mIoU = (IoULoss(inputs, targets_inv, smooth) + IoULoss(inputs, targets, smooth)) / 2
        else:
            mIoU = (weight[0] * IoULoss(inputs, targets_inv, smooth) + \
                    weight[1] * IoULoss(inputs, targets, smooth)) / weight.sum()

        return mIoU


# credits for Multiclass implementation to Kenneth Rithvik (in comments)
# https://www.kaggle.com/bigironsphere/loss-function-library-keras-pytorch/comments
class DiceLossMulticlass(nn.Module):
    def __init__(self, weight=None, reduction='mean'):
        super(mIoULoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs, targets of shapes (BATCH, NUM_CLASSES, H, W)

        # check the size of the weight
        if self.weight is not None:
            assert self.weight.shape == (targets.shape[1], )
        # make a copy not to change the default weight in the instance of DiceLossMulticlass
        weight = self.weight.copy()

        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        # flatten label and prediction images, leave BATCH and NUM_CLASSES
        # (BATCH, NUM_CLASSES, H, W) -> (BATCH, NUM_CLASSES, H * W)
        inputs = inputs.view(inputs.shape[0],inputs.shape[1],-1)
        targets = targets.view(targets.shape[0],targets.shape[1],-1)


        # get one number per each 2D image/mask pair
        # .sum(2) : (BATCH, NUM_CLASSES, H * W) -> (BATCH, NUM_CLASSES)
        intersection = (inputs * targets).sum(2)
        dice_coef = (2.*intersection + smooth)/(inputs.sum(2) + targets.sum(2) + smooth)
        dice_loss = 1 - dice_coef

        # no reduction, give a class mDiceLoss for every element in BATCH
        if reduction == 'none':
            # .mean(1) : (BATCH, NUM_CLASSES) -> (BATCH, )
            if weight is not None:
                return (dice_loss * weight).mean(1)
            else:
                return dice_loss.mean(1)

        # aggregate the loss for all elements in BATCH
        else:
            if reduction == 'mean':
                # .mean(0): (BATCH, NUM_CLASSES) -> (NUM_CLASSES, )
                dice_loss = dice_loss.mean(0)

            elif reduction == 'sum':
                # .sum(0): (BATCH, NUM_CLASSES) -> (NUM_CLASSES, )
                dice_loss = dice_loss.sum(0)
            else:
                raise ValueError("reduction should be one of 'none', 'mean', 'sum'")

            # (NUM_CLASSES, ) -> scalar
            if weight is not None:
                return (dice_loss*weight).mean()
            else:
                return dice_loss.mean()