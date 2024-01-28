import torch 
import torch.nn as nn
import torchvision
import numpy as np 

from utils import get_reg_loss, get_objectness_loss, get_cls_loss

class Loss(nn.Module):
    def __init__(self, reg_loss = "DIOU", obj_loss = "Focal", cls_loss = "BCE", alpha = 1.0, beta = 1.0, gamma = 1.0, C3OBJ = 4.0, C4OBJ = 1.0, C5OBJ = 1.0):
        super().__init__()
 
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.C3OBJ = C3OBJ
        self.C4OBJ = C4OBJ
        self.C5OBJ = C5OBJ

        self.BCE = get_objectness_loss(obj_loss)
        self.CE = get_cls_loss(cls_loss)
        self.regloss = get_reg_loss(reg_loss)
        
    def forward(self, pred, labels):

        C5_cls, C5_reg, C5_iou = pred[2]
        C4_cls, C4_reg, C4_iou = pred[1]
        C3_cls, C3_reg, C3_iou = pred[0]

        C3_cls_mask = labels[0]
        C4_cls_mask = labels[1]
        C5_cls_mask = labels[2]
        C3_iou_mask = labels[3]
        C4_iou_mask = labels[4]
        C5_iou_mask = labels[5]
        C3_reg_mask = labels[6]
        C4_reg_mask = labels[7]
        C5_reg_mask = labels[8]

        N_pos = C3_iou_mask.sum() + C4_iou_mask.sum() + C5_iou_mask.sum()
        
        # C3 classification loss
        #C3_cls_loss = self.CE(C3_cls, C3_cls_mask.squeeze(1).long())
        C3_cls_loss = self.CE(C3_cls, C3_cls_mask, C3_iou_mask)

        #C4 classification loss
        #C4_cls_loss = self.CE(C4_cls, C4_cls_mask.squeeze(1).long())
        C4_cls_loss = self.CE(C4_cls, C4_cls_mask, C4_iou_mask)

        #C5 classification loss
        #C5_cls_loss = self.CE(C5_cls, C5_cls_mask.squeeze(1).long())
        C5_cls_loss = self.CE(C5_cls, C5_cls_mask, C5_iou_mask)

        #cls_loss = (C3_cls_loss + C4_cls_loss + C5_cls_loss)/N_pos
        cls_loss = (C3_cls_loss + C4_cls_loss + C5_cls_loss)

        
        # C3 iou loss
        C3_iou_loss = self.BCE(C3_iou, C3_iou_mask)

        #C4 iou loss
        C4_iou_loss = self.BCE(C4_iou, C4_iou_mask)

        #C5 iou loss
        C5_iou_loss = self.BCE(C5_iou, C5_iou_mask)

        iou_loss = (self.C3OBJ*C3_iou_loss + self.C4OBJ*C4_iou_loss + self.C5OBJ*C5_iou_loss)
        
        # C3 reg loss
        C3_reg_loss =  (self.regloss(C3_reg, C3_reg_mask)*C3_iou_mask).sum()

        # C4 reg loss
        C4_reg_loss =  (self.regloss(C4_reg, C4_reg_mask)*C4_iou_mask).sum()

        # C5 reg loss
        C5_reg_loss =  (self.regloss(C5_reg, C5_reg_mask)*C5_iou_mask).sum()

        #reg_loss = (C3_reg_loss + C4_reg_loss + C5_reg_loss)/N_pos
        reg_loss = (C3_reg_loss + C4_reg_loss + C5_reg_loss)                
        
        #return ((self.alpha * iou_loss) + (self.beta * cls_loss) + (self.gamma * reg_loss), cls_loss, iou_loss, reg_loss, reg, npos)
        return ((self.alpha * iou_loss) + (self.beta * (cls_loss/N_pos)) + (self.gamma * (reg_loss/N_pos)), cls_loss, iou_loss, reg_loss, N_pos)

class LossExtended(nn.Module):
    def __init__(self, reg_loss = "DIOU", obj_loss = "Focal", cls_loss = "BCE", alpha = 1.0, beta = 1.0, gamma = 1.0, C3OBJ = 4.0, C4OBJ = 4.0, C5OBJ = 1.0, C6OBJ = 1.0):
        super().__init__()
 
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.C3OBJ = C3OBJ
        self.C4OBJ = C4OBJ
        self.C5OBJ = C5OBJ
        self.C6OBJ = C6OBJ

        self.BCE = get_objectness_loss(obj_loss)
        self.CE = get_cls_loss(cls_loss)
        self.regloss = get_reg_loss(reg_loss)
        
    def forward(self, pred, labels):

        C6_cls, C6_reg, C6_iou = pred[3]
        C5_cls, C5_reg, C5_iou = pred[2]
        C4_cls, C4_reg, C4_iou = pred[1]
        C3_cls, C3_reg, C3_iou = pred[0]

        C3_cls_mask = labels[0]
        C4_cls_mask = labels[1]
        C5_cls_mask = labels[2]
        C6_cls_mask = labels[3]
        C3_iou_mask = labels[4]
        C4_iou_mask = labels[5]
        C5_iou_mask = labels[6]
        C6_iou_mask = labels[7]
        C3_reg_mask = labels[8]
        C4_reg_mask = labels[9]
        C5_reg_mask = labels[10]
        C6_reg_mask = labels[11]

        N_pos = C3_iou_mask.sum() + C4_iou_mask.sum() + C5_iou_mask.sum() + C6_iou_mask.sum()
        
        # C3 classification loss
        #C3_cls_loss = self.CE(C3_cls, C3_cls_mask.squeeze(1).long())
        C3_cls_loss = self.CE(C3_cls, C3_cls_mask, C3_iou_mask)

        #C4 classification loss
        #C4_cls_loss = self.CE(C4_cls, C4_cls_mask.squeeze(1).long())
        C4_cls_loss = self.CE(C4_cls, C4_cls_mask, C4_iou_mask)

        #C5 classification loss
        #C5_cls_loss = self.CE(C5_cls, C5_cls_mask.squeeze(1).long())
        C5_cls_loss = self.CE(C5_cls, C5_cls_mask, C5_iou_mask)

        C6_cls_loss = self.CE(C6_cls, C6_cls_mask, C6_iou_mask)

        cls_loss = (C3_cls_loss + C4_cls_loss + C5_cls_loss + C6_cls_loss)

        # C3 iou loss
        C3_iou_loss = self.BCE(C3_iou, C3_iou_mask)

        #C4 iou loss
        C4_iou_loss = self.BCE(C4_iou, C4_iou_mask)

        #C5 iou loss
        C5_iou_loss = self.BCE(C5_iou, C5_iou_mask)

        C6_iou_loss = self.BCE(C6_iou, C6_iou_mask)

        iou_loss = (self.C3OBJ*C3_iou_loss + self.C4OBJ*C4_iou_loss + self.C5OBJ*C5_iou_loss + self.C6OBJ*C6_iou_loss)

        # C3 reg loss
        C3_reg_loss =  (self.regloss(C3_reg, C3_reg_mask)*C3_iou_mask).sum()

        # C4 reg loss
        C4_reg_loss =  (self.regloss(C4_reg, C4_reg_mask)*C4_iou_mask).sum()

        # C5 reg loss
        C5_reg_loss =  (self.regloss(C5_reg, C5_reg_mask)*C5_iou_mask).sum()

        C6_reg_loss =  (self.regloss(C6_reg, C6_reg_mask)*C6_iou_mask).sum()

        reg_loss = (C3_reg_loss + C4_reg_loss + C5_reg_loss + C6_reg_loss)
        
        return ((self.alpha * iou_loss) + (self.beta * (cls_loss/N_pos)) + (self.gamma * (reg_loss/N_pos)), cls_loss, iou_loss, reg_loss, N_pos)