import torch
import torch.nn as nn



class EBAM(nn.Module):
    '''
    Entropy Block Attention Module - https://arxiv.org/pdf/2206.03943.pdf
    '''

    def __init__(self, channels_size, reduction_ratio):
        super().__init__()

        self.channel = ChannelEBAM(channels_size, reduction_ratio)
        self.spatial = SpatialEBAM()
    
    def forward(self, x):
        x = self.channel(x)
        x = self.spatial(x)

        return x

class ChannelEBAM(nn.Module):
    def __init__(self, channels_size, reduction_ratio):
        super().__init__()

        self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.mlp = nn.Sequential(*[
            nn.Linear(channels_size, channels_size//reduction_ratio),
            nn.Linear(channels_size//reduction_ratio, channels_size)
        ])
    
    def forward(self, X):
        X_flatten = X.flatten(-2,-1)

        #Calculate prob
        X_prob = self.softmax(X_flatten)

        #Calculate entropy
        entropy = -(X_prob * torch.log2(X_prob)).sum(-1)
        entropy_attention = self.sigm(self.mlp(entropy)).unsqueeze(-1).unsqueeze(-1)

        return X * entropy_attention

class SpatialEBAM(nn.Module):
    def __init__(self):
        super().__init__()

        self.sigm = nn.Sigmoid()
        self.softmax = nn.Softmax(-1)
        self.conv = nn.Conv2d(1, 1, kernel_size = 7, stride = 1, padding = 3)

    def forward(self, X):
        #Calculate spatial prob
        X_spatial_prob = self.softmax(X)

        #Calculate entropy
        entropy = -(X_spatial_prob*torch.log2(X_spatial_prob)).sum(1).unsqueeze(1)
        max_entropy = torch.max(entropy)
        entropy = 1-(entropy/max_entropy)

        spatial_attention = self.sigm(self.conv(entropy))

        return X * spatial_attention



