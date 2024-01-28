import torch
import torch.nn as nn
import sys
import os
import math

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import CONV




class SPP(nn.Module):
    '''
    Spatial Pyramid Pooling - https://arxiv.org/pdf/1612.01105.pdf

    Args:
    '''
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.input_conv = CONV(input_channels, input_channels//2, 1)
        self.pooling1 = nn.MaxPool2d(5, 1, padding=2)
        self.pooling2 = nn.MaxPool2d(9, 1, padding=4)
        self.pooling3 = nn.MaxPool2d(13, 1, padding=6)
        self.output_conv = CONV(input_channels*2, output_channels, 1)

    def forward(self, x):
        X = self.input_conv(x)
        maxpool1 = self.pooling1(X)
        maxpool2 = self.pooling2(X)
        maxpool3 = self.pooling3(X)
        X = torch.cat((X, maxpool1, maxpool2, maxpool3), dim=1)

        

        return self.output_conv(X)
    

class SPPF(nn.Module):
    '''
    SPPF like in YOLOv8

    Args:
    '''
    def __init__(self, input_channels, output_channels):
        super().__init__()

        self.input_conv = CONV(input_channels, input_channels//2, 1)
        self.pooling = nn.MaxPool2d(5, 1, padding=2)
        self.output_conv = CONV(input_channels*2, output_channels, 1)

    def forward(self, x):
        X = self.input_conv(x)
        maxpool1 = self.pooling(X)
        maxpool2 = self.pooling(maxpool1)
        maxpool3 = self.pooling(maxpool2)
        X = torch.cat((X, maxpool1, maxpool2, maxpool3), dim=1)
        return self.output_conv(X)


class MHSA(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=4, act="relu", batch_norm=True):
        super().__init__()
        self.d = d_model
        self.reduce_channels = CONV(input_dim, d_model, 1, 1, 0, act, batch_norm)
        self.raise_channels = CONV(d_model, input_dim, 1, 1, 0, act, batch_norm)
        # For input batch_size x 3 x 704 x 1280 -> last map spatial dims: 11 x 20
        self.pos_encoding = nn.Parameter(torch.rand(11*20, d_model))
        self.SA = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=512, batch_first=True)

    def forward(self, X):
        result = self.reduce_channels(X)
        result = result.flatten(-2, -1).permute(0, 2, 1)
        result = result + self.pos_encoding
        result = self.SA(result)
        result = result.permute(0, 2, 1)
        result = result.reshape(X.shape[0], self.d, 11, 20)
        result = self.raise_channels(result)

        return result


class MHSABasic(nn.Module):
    def __init__(self, input_dim, d_model=256, n_heads=4, act="relu", batch_norm=True):
        super().__init__()
        self.d = d_model
        self.reduce_channels = CONV(input_dim, d_model, 1, 1, 0, act, batch_norm)
        self.raise_channels = CONV(d_model, input_dim, 1, 1, 0, act, batch_norm)
        # For input batch_size x 3 x 704 x 1280 -> last map spatial dims: 11 x 20
        self.pos_encoding = nn.Parameter(torch.rand(22*40, d_model))
        self.SA = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=256, batch_first=True)

    def forward(self, X):
        result = self.reduce_channels(X)
        result = result.flatten(-2, -1).permute(0, 2, 1)
        result = result + self.pos_encoding
        result = self.SA(result)
        result = result.permute(0, 2, 1)
        result = result.reshape(X.shape[0], self.d, 22, 40)
        result = self.raise_channels(result)

        return result
    






        


