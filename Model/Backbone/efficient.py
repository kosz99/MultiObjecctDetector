import torch
import torch.nn as nn
import torchvision


class EfficientNet(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        
        if pretrained:
            self.model = torchvision.models.efficientnet_v2_s(weights="EfficientNet_V2_S_Weights.IMAGENET1K_V1")
        
        else:
            self.model = torchvision.models.efficientnet_v2_s(weights=None)
        
        
        self.depth_channels = [64, 160, 1280]
        del self.model.avgpool
        del self.model.classifier
    def forward(self, X):
        C3 = self.model.features[:4](X)
        C4 = self.model.features[4:6](C3)
        C5 = self.model.features[6:](C4)

        return [C3, C4, C5]

class EfficientNetSmall(nn.Module):
    def __init__(self, pretrained = False):
        super().__init__()
        
        if pretrained:
            self.model = torchvision.models.efficientnet_b3(weights='IMAGENET1K_V1')
        
        else:
            self.model = torchvision.models.efficientnet_b3(weights=None)
        
        
        self.depth_channels = [48, 136, 1536]
        #del self.model.avgpool
        #del self.model.classifier
    def forward(self, X):
        C3 = self.model.features[:4](X)
        C4 = self.model.features[4:6](C3)
        C5 = self.model.features[6:](C4)

        return [C3, C4, C5] 


