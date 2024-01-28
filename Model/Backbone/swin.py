import torch
import torch.nn as nn
import torchvision


class Swin(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        
        if pretrained:
            self.model = torchvision.models.swin_v2_t(weights="Swin_V2_T_Weights.IMAGENET1K_V1")
        
        else:
            self.model = torchvision.models.swin_v2_t(weights=None)
        
        
        self.depth_channels = [192, 384, 768]
        
    def forward(self, X):
        C3 = self.model.features[:4](X)
        C4 = self.model.features[4:6](C3)
        C5 = self.model.features[6:](C4)

        return [C3.permute(0,3,1,2), C4.permute(0,3,1,2), C5.permute(0,3,1,2)]






