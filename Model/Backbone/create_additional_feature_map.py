import torch
import torch.nn as nn
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import CONV, C3_Module
from Backbone.efficient import EfficientNetSmall, EfficientNet
from Backbone.swin import Swin
from Backbone.regnet import RegNet

class CreateAdditionalFeatureMapBackbone(nn.Module):
    def __init__(self, name, pretrained=False, batch_norm = True, act = "relu", bottleneck_num = 3, bottleneck_shortcut = True):
        super().__init__()
        self.model = self.create_backbone(name, pretrained=pretrained)
        self.spatial_shrink = CONV(self.model.depth_channels[-1], self.model.depth_channels[-1], kernel_size=3, stride=2, padding=1, act=act, batch_norm=batch_norm)
        self.C3 = C3_Module(self.model.depth_channels[-1], self.model.depth_channels[-1], self.model.depth_channels[-1], act, batch_norm, bottleneck_shortcut, bottleneck_num=bottleneck_num)

    def create_backbone(self, name = "EfficientNetSmall", pretrained = False):
        assert name == "EfficientNetSmall" or name == "EfficientNet" or name == "Swin" or name == "RegNet", f"unknown backbone type"
        backbone = {
            "EfficientNet" : EfficientNet(pretrained=pretrained),
            "EfficientNetSmall" : EfficientNetSmall(pretrained=pretrained),
            "Swin" : Swin(pretrained=pretrained),
            "RegNet" : RegNet(pretrained=pretrained)
        }
        return backbone[name]        

    def forward(self, X):
        model_output = self.model(X)
        last_map = self.spatial_shrink(model_output[-1])
        last_map = self.C3(last_map)

        return [model_output[0], model_output[1], model_output[2], last_map]
    

class SimpleCreateAdditionalFeatureMapBackbone(nn.Module):
    def __init__(self, name, pretrained=False, batch_norm = True, act = "relu"):
        super().__init__()
        self.model = self.create_backbone(name, pretrained=pretrained)
        self.spatial_shrink = CONV(self.model.depth_channels[-1], self.model.depth_channels[-1], kernel_size=3, stride=2, padding=1, act=act, batch_norm=batch_norm)

    def create_backbone(self, name = "EfficientNetSmall", pretrained = False):
        assert name == "EfficientNetSmall" or name == "EfficientNet" or name == "Swin" or name == "RegNet", f"unknown backbone type"
        backbone = {
            "EfficientNet" : EfficientNet(pretrained=pretrained),
            "EfficientNetSmall" : EfficientNetSmall(pretrained=pretrained),
            "Swin" : Swin(pretrained=pretrained),
            "RegNet" : RegNet(pretrained=pretrained)
        }
        return backbone[name]        

    def forward(self, X):
        model_output = self.model(X)
        last_map = self.spatial_shrink(model_output[-1])

        return [model_output[0], model_output[1], model_output[2], last_map]
    

