import torch
import torch.nn as nn


class CBAM(nn.Module):
    '''
    Convolutional Block Attention Module - https://arxiv.org/pdf/1807.06521.pdf
    '''
    def __init__(self, channels_size, reduction_ratio=2):
        super().__init__()

        self.channelAttentionModule = ChannelAttentionModule(channels_size, reduction_ratio)
        self.spatialAttentionModule = SpatialAttentionModule(channels_size)

    def forward(self, x):
        x = self.channelAttentionModule(x)
        x = self.spatialAttentionModule(x)

        return x

class ChannelAttentionModule(nn.Module):
    def __init__(self, channels_size, reduction_ratio):
        super().__init__()

        '''
        self.max_pooling = nn.MaxPool2d((input_shape[-2], input_shape[-1]))
        self.avg_pooling = nn.AvgPool2d((input_shape[-2], input_shape[-1]))
        '''
        self.max_pooling = nn.AdaptiveMaxPool2d((1,1))
        self.avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.mlp = nn.Sequential(*[
            nn.Linear(channels_size, channels_size//reduction_ratio),
            nn.ReLU(),
            nn.Linear(channels_size//reduction_ratio, channels_size)    
        ])

        self.act = nn.Sigmoid()
    
    def forward(self, x):
        MaxPool = self.mlp(self.max_pooling(x).movedim(-3, -1)).movedim(-1, -3)
        AvgPool = self.mlp(self.avg_pooling(x).movedim(-3, -1)).movedim(-1, -3)
        
        attention  = self.act(MaxPool+AvgPool)

        return x * attention
        

class SpatialAttentionModule(nn.Module):

    def __init__(self, channels_size):
        super().__init__()

        self.max_pooling = nn.MaxPool3d((channels_size, 1, 1))
        self.avg_pooling = nn.AvgPool3d((channels_size, 1, 1))

        self.conv = nn.Conv2d(2, 1, kernel_size= 7, stride= 1, padding= 3)

        self.act = nn.Sigmoid()

    def forward(self, x):
        cat = torch.cat((
            self.max_pooling(x),
            self.avg_pooling(x)
        ), dim = 1)
        
        attention = self.act(self.conv(cat))

        return x * attention

