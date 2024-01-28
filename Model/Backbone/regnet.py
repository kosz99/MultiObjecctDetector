import torch
import torchvision
import torch.nn as nn

# Regnet paper: https://arxiv.org/pdf/2101.00590.pdf


class RegNet(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        if pretrained == True:
            self.model = torchvision.models.regnet_x_3_2gf(weights='DEFAULT')
        else:
            self.model = torchvision.models.regnet_x_3_2gf(weights= None)
        
        self.depth_channels = [192, 432, 1008]
        del self.model.avgpool
        del self.model.fc
    
    def forward(self, X):
        X = self.model.stem(X)
        C3 = self.model.trunk_output[:2](X)   #feature map shape: Batch_size x 192 x input_H / 8 x input_W / 8
        C4 = self.model.trunk_output[2](C3)   #feature map shape: Batch_size x 432 x input_H / 16 x input_W / 16
        C5 = self.model.trunk_output[3](C4)   #feature map shape: Batch_size x 1008 x input_H / 32 x input_W / 32
        
        return [C3, C4, C5]