import torch
import torch.nn as nn
import os
import sys
import time




SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from cbam import CBAM
from utils import get_upsample, C3_Module, CONV, GSCONV, VoVGSCSP

class CSPPANExtended(nn.Module):
    '''
    CSP-PAN neck - https://docs.ultralytics.com/yolov5/architecture/
    '''
    def __init__(self, backbone_output_depth_channels, channels, act = "relu", batch_norm = False, bottleneck_shortcut = False, bottleneck_num = 3):
        super().__init__()

        self.channel_alignmentC3 =  CONV(backbone_output_depth_channels[0], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC4 =  CONV(backbone_output_depth_channels[1], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC5 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC6 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)       

        self.upsample = get_upsample()

        self.C3Block_C5 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_C4 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_C3 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_P3 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_P4 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_P5 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)

        self.spatial_reductionP3 = CONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP4 = CONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP5 = CONV(channels, channels, 3, 2, 1, act, batch_norm)    

    def forward(self, X):
        C3, C4, C5, C6 = X

        C3 = self.channel_alignmentC3(C3)
        C4 = self.channel_alignmentC4(C4)
        C5 = self.channel_alignmentC5(C5)
        C6 = self.channel_alignmentC6(C6)

        C6_prim = self.upsample(C6)
        C5 = torch.cat((C5, C6_prim), dim=1)
        C5 = self.C3Block_C5(C5)
        
        C5_prim = self.upsample(C5)
        C4 = torch.cat((C4, C5_prim), dim=1)
        C4 = self.C3Block_C4(C4)

        C4_prim = self.upsample(C4)
        C3 = torch.cat((C3, C4_prim), dim=1)
        P3 = self.C3Block_C3(C3)

        P3_prim = self.spatial_reductionP3(P3)
        P3_prim = torch.cat((P3_prim, C4), dim=1)
        P4 = self.C3Block_P3(P3_prim)

        P4_prim = self.spatial_reductionP4(P4)
        P4_prim = torch.cat((P4_prim, C5), dim=1)
        P5 = self.C3Block_P4(P4_prim)

        P5_prim = self.spatial_reductionP5(P5)
        P5_prim = torch.cat((P5_prim, C6), dim=1)
        P6 = self.C3Block_P5(P5_prim)


        return [P3, P4, P5, P6]

class GSCSPPANExtended(nn.Module):
    '''
    CSP-PAN neck - https://docs.ultralytics.com/yolov5/architecture/
    '''
    def __init__(self, backbone_output_depth_channels, channels, act = "relu", batch_norm = False, bottleneck_num = 1):
        super().__init__()

        self.channel_alignmentC3 =  CONV(backbone_output_depth_channels[0], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC4 =  CONV(backbone_output_depth_channels[1], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC5 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC6 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)       

        self.upsample = get_upsample()

        self.C3Block_C5 = VoVGSCSP(2*channels, channels, channels, act, batch_norm, bottleneck_num)
        self.C3Block_C4 = VoVGSCSP(2*channels, channels, channels, act, batch_norm, bottleneck_num)
        self.C3Block_C3 = VoVGSCSP(2*channels, channels, channels, act, batch_norm, bottleneck_num)
        self.C3Block_P3 = VoVGSCSP(2*channels, channels, channels, act, batch_norm, bottleneck_num)
        self.C3Block_P4 = VoVGSCSP(2*channels, channels, channels, act, batch_norm, bottleneck_num)
        self.C3Block_P5 = VoVGSCSP(2*channels, channels, channels, act, batch_norm, bottleneck_num)

        self.spatial_reductionP3 = GSCONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP4 = GSCONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP5 = GSCONV(channels, channels, 3, 2, 1, act, batch_norm)    

    def forward(self, X):
        C3, C4, C5, C6 = X

        C3 = self.channel_alignmentC3(C3)
        C4 = self.channel_alignmentC4(C4)
        C5 = self.channel_alignmentC5(C5)
        C6 = self.channel_alignmentC6(C6)

        C6_prim = self.upsample(C6)
        C5 = torch.cat((C5, C6_prim), dim=1)
        C5 = self.C3Block_C5(C5)
        
        C5_prim = self.upsample(C5)
        C4 = torch.cat((C4, C5_prim), dim=1)
        C4 = self.C3Block_C4(C4)

        C4_prim = self.upsample(C4)
        C3 = torch.cat((C3, C4_prim), dim=1)
        P3 = self.C3Block_C3(C3)

        P3_prim = self.spatial_reductionP3(P3)
        P3_prim = torch.cat((P3_prim, C4), dim=1)
        P4 = self.C3Block_P3(P3_prim)

        P4_prim = self.spatial_reductionP4(P4)
        P4_prim = torch.cat((P4_prim, C5), dim=1)
        P5 = self.C3Block_P4(P4_prim)

        P5_prim = self.spatial_reductionP5(P5)
        P5_prim = torch.cat((P5_prim, C6), dim=1)
        P6 = self.C3Block_P5(P5_prim)


        return [P3, P4, P5, P6]

class FPNPANExtended(nn.Module):
    def __init__(self, backbone_output_depth_channels, channels, act = "relu", batch_norm = False, neck_fusion_type = "cat"):
        super().__init__()
        self.neck_fusion_type = neck_fusion_type
        self.channel_alignmentC3 =  CONV(backbone_output_depth_channels[0], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC4 =  CONV(backbone_output_depth_channels[1], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC5 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC6 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)
        
        self.upsample = get_upsample()

        self.spatial_reductionP3 = CONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP4 = CONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP5 = CONV(channels, channels, 3, 2, 1, act, batch_norm)

        if self.neck_fusion_type == "cat":
            self.C3Block_C5 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
            self.C3Block_C4 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
            self.C3Block_C3 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
            self.C3Block_P3 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
            self.C3Block_P4 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
            self.C3Block_P5 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
        else:
            self.C3Block_C5 = CONV(channels, channels, 3, 1, 1, act, batch_norm)
            self.C3Block_C4 = CONV(channels, channels, 3, 1, 1, act, batch_norm)
            self.C3Block_C3 = CONV(channels, channels, 3, 1, 1, act, batch_norm)
            self.C3Block_P3 = CONV(channels, channels, 3, 1, 1, act, batch_norm)
            self.C3Block_P4 = CONV(channels, channels, 3, 1, 1, act, batch_norm) 
            self.C3Block_P5 = CONV(channels, channels, 3, 1, 1, act, batch_norm)           

    def forward(self, X):

        C3, C4, C5, C6 = X

        C3 = self.channel_alignmentC3(C3)
        C4 = self.channel_alignmentC4(C4)
        C5 = self.channel_alignmentC5(C5)
        C6 = self.channel_alignmentC5(C6)

        C6_prim = self.upsample(C6)
        if self.neck_fusion_type == "cat":
            C5 = torch.cat((C5, C6_prim), dim=1)
        elif self.neck_fusion_type == "add":
            C5 = C5+C6_prim
        C5 = self.C3Block_C5(C5)

        C5_prim = self.upsample(C5)
        if self.neck_fusion_type == "cat":
            C4 = torch.cat((C4, C5_prim), dim=1)
        elif self.neck_fusion_type == "add":
            C4 = C4+C5_prim
        C4 = self.C3Block_C4(C4)

        C4_prim = self.upsample(C4)
        if self.neck_fusion_type == "cat":
            C3 = torch.cat((C3, C4_prim), dim=1)
        elif self.neck_fusion_type == "add":
            C3 = C3+C4_prim
        P3 = self.C3Block_C3(C3)

        P3_prim = self.spatial_reductionP3(P3)
        if self.neck_fusion_type == "cat":
            P3_prim = torch.cat((P3_prim, C4), dim=1)
        elif self.neck_fusion_type == "add":
            P3_prim = P3_prim+C4
        P4 = self.C3Block_P3(P3_prim)

        P4_prim = self.spatial_reductionP4(P4)
        if self.neck_fusion_type == "cat":
            P4_prim = torch.cat((P4_prim, C5), dim=1)
        elif self.neck_fusion_type == "add":
            P4_prim = P4_prim + C5
        P5 = self.C3Block_P4(P4_prim)

        P5_prim = self.spatial_reductionP5(P5)
        if self.neck_fusion_type == "cat":
            P5_prim = torch.cat((P5_prim, C6), dim=1)
        elif self.neck_fusion_type == "add":
            P5_prim = P5_prim + C6
        P6 = self.C3Block_P5(P5_prim)


        return [P3, P4, P5, P6]

class FPNPAN_CBAMExtended(nn.Module):
    def __init__(self, backbone_output_depth_channels, channels, act = "relu", batch_norm = False, reduction_ratio = 2):
        super().__init__()
        self.channel_alignmentC3 =  CONV(backbone_output_depth_channels[0], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC4 =  CONV(backbone_output_depth_channels[1], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC5 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC6 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)

        self.upsample = get_upsample()

        self.spatial_reductionP3 = CONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP4 = CONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP5 = CONV(channels, channels, 3, 2, 1, act, batch_norm)

        self.C3Block_C5 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
        self.C3Block_C4 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
        self.C3Block_C3 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
        self.C3Block_P3 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
        self.C3Block_P4 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)
        self.C3Block_P5 = CONV(2*channels, channels, 3, 1, 1, act, batch_norm)

        self.CBAM_C5 = CBAM(channels*2, reduction_ratio)
        self.CBAM_C4 = CBAM(channels*2, reduction_ratio)
        self.CBAM_C3 = CBAM(channels*2, reduction_ratio)
        self.CBAM_P3 = CBAM(channels*2, reduction_ratio)
        self.CBAM_P4 = CBAM(channels*2, reduction_ratio)
        self.CBAM_P5 = CBAM(channels*2, reduction_ratio)
    
    def forward(self, X):
        C3, C4, C5, C6 = X

        C3 = self.channel_alignmentC3(C3)
        C4 = self.channel_alignmentC4(C4)
        C5 = self.channel_alignmentC5(C5)
        C6 = self.channel_alignmentC6(C6)

        C6_prim = self.upsample(C6)
        C5 = torch.cat((C5, C6_prim), dim=1)
        C5 = self.CBAM_C5(C5)
        C5 = self.C3Block_C5(C5)

        C5_prim = self.upsample(C5)
        C4 = torch.cat((C4, C5_prim), dim=1)
        C4 = self.CBAM_C4(C4)
        C4 = self.C3Block_C4(C4)

        C4_prim = self.upsample(C4)
        C3 = torch.cat((C3, C4_prim), dim=1)
        C3 = self.CBAM_C3(C3)
        P3 = self.C3Block_C3(C3)

        P3_prim = self.spatial_reductionP3(P3)
        P3_prim = torch.cat((P3_prim, C4), dim=1)
        P3_prim = self.CBAM_P3(P3_prim)
        P4 = self.C3Block_P3(P3_prim)

        P4_prim = self.spatial_reductionP4(P4)
        P4_prim = torch.cat((P4_prim, C5), dim=1)
        P4_prim = self.CBAM_P4(P4_prim)
        P5 = self.C3Block_P4(P4_prim)

        P5_prim = self.spatial_reductionP5(P5)
        P5_prim = torch.cat((P5_prim, C6), dim=1)
        P5_prim = self.CBAM_P5(P5_prim)
        P6 = self.C3Block_P5(P5_prim)

        return [P3, P4, P5, P6]


class CSPPAN_CBAMExtended(nn.Module):
    '''
    CSP-PAN neck - https://docs.ultralytics.com/yolov5/architecture/
    '''
    def __init__(self, backbone_output_depth_channels, channels, act = "relu", batch_norm = False, bottleneck_shortcut = False, bottleneck_num = 3, reduction_ratio = 2):
        super().__init__()

        self.channel_alignmentC3 =  CONV(backbone_output_depth_channels[0], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC4 =  CONV(backbone_output_depth_channels[1], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC5 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)
        self.channel_alignmentC6 =  CONV(backbone_output_depth_channels[2], channels, 1, 1, 0, act, batch_norm)       

        self.upsample = get_upsample()

        self.C3Block_C5 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_C4 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_C3 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_P3 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_P4 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)
        self.C3Block_P5 = C3_Module(2*channels, channels, channels, act, batch_norm, bottleneck_shortcut, bottleneck_num)

        self.spatial_reductionP3 = CONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP4 = CONV(channels, channels, 3, 2, 1, act, batch_norm)
        self.spatial_reductionP5 = CONV(channels, channels, 3, 2, 1, act, batch_norm)

        self.CBAM_C5 = CBAM(channels*2, reduction_ratio)
        self.CBAM_C4 = CBAM(channels*2, reduction_ratio)
        self.CBAM_C3 = CBAM(channels*2, reduction_ratio)
        self.CBAM_P3 = CBAM(channels*2, reduction_ratio)
        self.CBAM_P4 = CBAM(channels*2, reduction_ratio)
        self.CBAM_P5 = CBAM(channels*2, reduction_ratio)    

    def forward(self, X):
        C3, C4, C5, C6 = X

        C3 = self.channel_alignmentC3(C3)
        C4 = self.channel_alignmentC4(C4)
        C5 = self.channel_alignmentC5(C5)
        C6 = self.channel_alignmentC6(C6)

        C6_prim = self.upsample(C6)
        C5 = torch.cat((C5, C6_prim), dim=1)
        C5 = self.CBAM_C5(C5)
        C5 = self.C3Block_C5(C5)
        
        C5_prim = self.upsample(C5)
        C4 = torch.cat((C4, C5_prim), dim=1)
        C4 = self.CBAM_C5(C4)
        C4 = self.C3Block_C4(C4)

        C4_prim = self.upsample(C4)
        C3 = torch.cat((C3, C4_prim), dim=1)
        C3 = self.CBAM_C5(C3)
        P3 = self.C3Block_C3(C3)

        P3_prim = self.spatial_reductionP3(P3)
        P3_prim = torch.cat((P3_prim, C4), dim=1)
        P3_prim = self.CBAM_P3(P3_prim)
        P4 = self.C3Block_P3(P3_prim)

        P4_prim = self.spatial_reductionP4(P4)
        P4_prim = torch.cat((P4_prim, C5), dim=1)
        P4_prim = self.CBAM_P4(P4_prim)
        P5 = self.C3Block_P4(P4_prim)

        P5_prim = self.spatial_reductionP5(P5)
        P5_prim = torch.cat((P5_prim, C6), dim=1)
        P5_prim = self.CBAM_P5(P5_prim)
        P6 = self.C3Block_P5(P5_prim)


        return [P3, P4, P5, P6]

