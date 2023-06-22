""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None ,use_selu=False):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if use_selu:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.SELU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.SELU(inplace=True)
            )
            
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        


    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True,use_selu=False):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2, use_selu=use_selu)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels, use_selu=use_selu)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


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
        
        
        
        
        
        
class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
            
    def forward(self, x):
        x_avg_pool = torch.mean(x,1).unsqueeze(1)
        x_max_pool = torch.max(x,1)[0].unsqueeze(1)
        attention = torch.cat((x_avg_pool, x_max_pool), dim=1) # (batch, 2, 224, 224)
        attention = self.conv(attention) # (batch, 1, 224, 224)
        return x*attention