import torch
import torch.nn as nn
import torchvision
#from model_size import model_size

from Backbone.efficient import EfficientNet, EfficientNetSmall
from Backbone.swin import Swin
from Backbone.regnet import RegNet
from Backbone.create_additional_feature_map import CreateAdditionalFeatureMapBackbone, SimpleCreateAdditionalFeatureMapBackbone
from Backbone.spp import SPP, SPPF, MHSA, MHSABasic
from Neck.neck import FPNPAN, CSPPAN, FPNPAN_CBAM, CSPPAN_CBAM, GSCSPPAN
from Neck.neck_additional_scale import CSPPANExtended, FPNPAN_CBAMExtended, FPNPANExtended, CSPPAN_CBAMExtended, GSCSPPANExtended
from Head.head import Head, BaseHead, ManyHeads, ManyHeadsExtended



class Model(nn.Module):
    def __init__(
            self,
            backbone_name = "EfficientNet",
            pretrained_backbone = False, 
            spp_exist = True, 
            spp_type = "SPPF", 
            channels = 256,
            neck_name = "CSPPAN",
            neck_act = "relu",
            neck_batch_norm = True, 
            reduction_ratio = 2, 
            neck_fusion_type = "cat", 
            bottleneck_shortcut = False, 
            bottleneck_num = 3,
            class_number = 80, 
            head_type = "shared", 
            head_activation_function = "relu",
            head_batch_norm = False, 
            head_num_convs_blocks = 4, 
            ):
        super().__init__()
        self.backbone = self.create_backbone(backbone_name, pretrained_backbone)
        self.spp_exist = spp_exist
        if self.spp_exist:
            self.spp = self.create_spp(spp_type, neck_act, neck_batch_norm)
        self.neck = self.create_neck(self.backbone.depth_channels, channels, neck_name, neck_act, neck_batch_norm, reduction_ratio, neck_fusion_type, bottleneck_shortcut, bottleneck_num)
        self.head = self.create_head(channels, class_number, head_type, head_activation_function, head_batch_norm, head_num_convs_blocks)

    def forward(self, X):
        x = self.backbone(X)
        if self.spp_exist:
            x[-1] = self.spp(x[-1])
        x = self.neck(x)
        x = self.head(x)
        return x
    
    def create_backbone(self, name = "EfficientNetSmall", pretrained = False):
        assert name == "EfficientNetSmall" or name == "EfficientNet" or name == "Swin" or name == "RegNet", f"unknown backbone type"
        backbone = {
            "EfficientNet" : EfficientNet(pretrained=pretrained),
            "EfficientNetSmall" : EfficientNetSmall(pretrained=pretrained),
            "Swin" : Swin(pretrained=pretrained),
            "RegNet" : RegNet(pretrained=pretrained)
        }
        return backbone[name]
    
    def create_spp(self, name = "SPPF", act = "relu", batch_norm = True):
        assert name == "SPP" or name == "SPPF" or name == "MHSA", f"unknown spp type"
        spp = {
            "SPP" : SPP(self.backbone.depth_channels[-1], self.backbone.depth_channels[-1]),
            "SPPF" : SPPF(self.backbone.depth_channels[-1], self.backbone.depth_channels[-1]),
            "MHSA" : MHSABasic(self.backbone.depth_channels[-1], 256, 4, act, batch_norm)
        }
        return spp[name]

    def create_neck(self, backbone_output_depth_channels, channels, neck_name = "CSPPAN", act = "relu", batch_norm = False, reduction_ratio = 2, neck_fusion_type = "cat", bottleneck_shortcut = False, bottleneck_num = 3):
        assert neck_name == "FPNPAN" or neck_name == "CSPPAN" or neck_name == "FPNPAN_CBAM" or neck_name == "CSPPAN_CBAM" or neck_name == "GSCSPPAN", f"unknown neck type"
        neck = {
            "FPNPAN" : FPNPAN(backbone_output_depth_channels, channels, act, batch_norm, neck_fusion_type),
            "CSPPAN" : CSPPAN(backbone_output_depth_channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num),
            "GSCSPPAN" : GSCSPPAN(backbone_output_depth_channels, channels, act, batch_norm, bottleneck_num),            
            "FPNPAN_CBAM" : FPNPAN_CBAM(backbone_output_depth_channels, channels, act, batch_norm, reduction_ratio),
            "CSPPAN_CBAM" : CSPPAN_CBAM(backbone_output_depth_channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num, reduction_ratio)
        }
        return neck[neck_name]
     
    def create_head(self, channels, class_number, head_type = "shared", activation_function = "relu", batch_norm = False, num_convs_blocks = 4):
        assert head_type == "shared" or head_type == "seperated", f"unknown head type"
        stride_params = nn.ParameterList([nn.Parameter((torch.randint(0,10,(1,))).float()),
                                               nn.Parameter((torch.randint(0,10,(1,))).float()),
                                               nn.Parameter((torch.randint(0,10,(1,))).float())])
        head = {
            "shared" : Head(stride_params, channels, class_number, activation_function, batch_norm, num_convs_blocks),
            "seperated" : ManyHeads(stride_params, channels, class_number, activation_function, batch_norm, num_convs_blocks)
        }
        return head[head_type]

class ModelExtended(nn.Module):
    def __init__(
            self,
            backbone_name = "EfficientNetSmall",
            pretrained_backbone = False, 
            spp_exist = True, 
            spp_type = "SPPF", 
            channels = 256,
            neck_name = "CSPPAN",
            neck_act = "relu",
            neck_batch_norm = True, 
            reduction_ratio = 2, 
            neck_fusion_type = "cat", 
            bottleneck_shortcut = False, 
            bottleneck_num = 3,
            class_number = 80, 
            head_type = "shared", 
            head_activation_function = "relu",
            head_batch_norm = False, 
            head_num_convs_blocks = 4
            ):
        super().__init__()
        self.backbone = self.create_backbone(backbone_name, pretrained_backbone, neck_batch_norm, neck_act)
        self.spp_exist = spp_exist
        if self.spp_exist:
            self.spp = self.create_spp(spp_type, neck_act, neck_batch_norm)
        self.neck = self.create_neck(self.backbone.model.depth_channels, channels, neck_name, neck_act, neck_batch_norm, reduction_ratio, neck_fusion_type, bottleneck_shortcut, bottleneck_num)
        self.head = self.create_head(channels, class_number, head_type, head_activation_function, head_batch_norm, head_num_convs_blocks)

    def forward(self, X):
        x = self.backbone(X)
        if self.spp_exist:
            x[-1] = self.spp(x[-1])
        x = self.neck(x)
        x = self.head(x)
        return x
    
    def create_backbone(self, name = "EfficientNetSmall", pretrained = False, batch_norm = True, act = "relu"):
        assert name == "EfficientNetSmall" or name == "EfficientNet" or name == "Swin" or "RegNet", f"unknown backbone type"
        backbone = {
            "EfficientNet" : SimpleCreateAdditionalFeatureMapBackbone("EfficientNet", pretrained, batch_norm, act),
            "EfficientNetSmall" : SimpleCreateAdditionalFeatureMapBackbone("EfficientNetSmall", pretrained, batch_norm, act),
            "Swin" : SimpleCreateAdditionalFeatureMapBackbone("Swin", pretrained, batch_norm, act),
            "RegNet" : SimpleCreateAdditionalFeatureMapBackbone("RegNet", pretrained, batch_norm, act)
        }
        return backbone[name]
    
    def create_spp(self, name = "SPPF", act = "relu", batch_norm = True):
        assert name == "SPP" or name == "SPPF" or name == "MHSA", f"unknown spp type"
        spp = {
            "SPP" : SPP(self.backbone.model.depth_channels[-1], self.backbone.model.depth_channels[-1]),
            "SPPF" : SPPF(self.backbone.model.depth_channels[-1], self.backbone.model.depth_channels[-1]),
            "MHSA" : MHSA(self.backbone.model.depth_channels[-1], 256, 4, act, batch_norm)
        }
        return spp[name]

    def create_neck(self, backbone_output_depth_channels, channels, neck_name = "CSPPAN", act = "relu", batch_norm = False, reduction_ratio = 2, neck_fusion_type = "cat", bottleneck_shortcut = False, bottleneck_num = 3):
        assert neck_name == "FPNPAN" or neck_name == "CSPPAN" or neck_name == "FPNPANCBAM" or neck_name == "CSPPANCBAM" or neck_name == "GSCSPPAN", f"unknown neck type"
        neck = {
            "CSPPAN" : CSPPANExtended(backbone_output_depth_channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num),
            "FPNPAN" : FPNPANExtended(backbone_output_depth_channels, channels, act, batch_norm, neck_fusion_type),
            "GSCSPPAN" : GSCSPPANExtended(backbone_output_depth_channels, channels, act, batch_norm, bottleneck_num),
            "FPNPANCBAM" : FPNPAN_CBAMExtended(backbone_output_depth_channels, channels, act, batch_norm, reduction_ratio),
            "CSPPANCBAM" : CSPPAN_CBAMExtended(backbone_output_depth_channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num, reduction_ratio)            
        }
        return neck[neck_name]
     
    def create_head(self, channels, class_number, head_type = "shared", activation_function = "relu", batch_norm = False, num_convs_blocks = 4):
        assert head_type == "shared" or head_type == "seperated", f"unknown head type"
        stride_params = nn.ParameterList([nn.Parameter((torch.randint(0,10,(1,))).float()),
                                               nn.Parameter((torch.randint(0,10,(1,))).float()),
                                               nn.Parameter((torch.randint(0,10,(1,))).float()),
                                               nn.Parameter((torch.randint(0,10,(1,))).float())])
        head = {
            "shared" : Head(stride_params, channels, class_number, activation_function, batch_norm, num_convs_blocks),
            "seperated" : ManyHeadsExtended(stride_params, channels, class_number, activation_function, batch_norm, num_convs_blocks)
        }
        return head[head_type]
