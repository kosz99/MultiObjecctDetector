import torch
import torch.nn as nn
import torchvision


class MaxVIT(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        
        if pretrained:
            self.model = torchvision.models.maxvit_t(weights='MaxVit_T_Weights.IMAGENET1K_V1')
        
        else:
            self.model = torchvision.models.maxvit_t(weights=None)
        
        
        self.depth_channels = [128, 256, 512]
        del self.model.classifier
    def forward(self, X):
        X = self.model.stem(X)
        X = self.model.blocks[0](X)
        C3 = self.model.blocks[1](X)
        C4 = self.model.blocks[2](C3)
        C5 = self.model.blocks[3](C4)

        return C3, C4, C5 


