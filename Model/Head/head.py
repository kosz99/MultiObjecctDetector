import torch
import torch.nn as nn
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from utils import CONV

class Head(nn.Module):
    '''
    None anchor box object detector head - same head for each neck output

    Head generates 3 branches: IOU (objectness) 1xWxH, Reg 4xWxH (l,r,t,b), cls #clsxWxH

    Args:
        scalar (List[int]): list of parameters for each reg output level like FCOS
        internal_dim (int): number of channels inside Head
        class_number (int): number of classes
        activation (str): activation function
        batch_norm (bool): bool
        num_convs_blocks (int): number of CONV blocks
    '''
    def __init__(self, scalar, internal_dim, class_number, activation_function = "relu", batch_norm = False, num_convs_blocks = 4):
        super().__init__()

        self.scalar = scalar
        self.internal_dim = internal_dim
        self.class_number = class_number
        self.batch_norm = batch_norm
        

        self.reg = nn.Conv2d(self.internal_dim, 4, 3, 1, 1)
        self.iou = nn.Conv2d(self.internal_dim, 1, 3, 1, 1)
        self.relu = nn.ReLU()


        self.cls = nn.Sequential(*[CONV(self.internal_dim, self.internal_dim, 3, 1, 1, activation_function, self.batch_norm) for _ in range(num_convs_blocks)],
                                   nn.Conv2d(self.internal_dim, class_number, 3, 1, 1))

        self.reg_iou = nn.Sequential(*[CONV(self.internal_dim, self.internal_dim, 3, 1, 1, activation_function, self.batch_norm) for _ in range(num_convs_blocks)])
    def forward(self, X):
        '''
        Args:
            X (List[Tensor]): FPN/FPNPAN output - [N3, N4, N5]
            
        
        Returns:
            results (List[Tensor]): list of cls_output, reg_output, iou_output for each level 
        '''

        output = []
        for idx, x in enumerate(X):
            cls_output = self.cls(x)
            reg_iou = self.reg_iou(x)
            reg_output = self.relu(self.reg(reg_iou)*self.scalar[idx])
            iou_output = self.iou(reg_iou)
            output.append((cls_output, reg_output, iou_output))

        return output
    
class BaseHead(nn.Module):
    '''
    None anchor box object detector head 

    Head generates 3 branches: IOU (objectness) 1xWxH, Reg 4xWxH (l,r,t,b), cls #clsxWxH

    Args:
        internal_dim (int): number of channels inside Head
        class_number (int): number of classes
        activation (str): activation function
        batch_norm (bool): bool
        num_convs_blocks (int): number of CONV blocks
    '''
    def __init__(self, internal_dim, class_number, activation_function = "relu", batch_norm = False, num_convs_blocks = 4):
        super().__init__()

        self.internal_dim = internal_dim
        self.class_number = class_number
        self.batch_norm = batch_norm

        self.reg = nn.Conv2d(self.internal_dim, 4, 3, 1, 1)
        self.iou = nn.Conv2d(self.internal_dim, 1, 3, 1, 1)
        self.relu = nn.ReLU()

        self.cls = nn.Sequential(*[CONV(self.internal_dim, self.internal_dim, 3, 1, 1, activation_function, self.batch_norm) for _ in range(num_convs_blocks)],
                                   nn.Conv2d(self.internal_dim, class_number, 3, 1, 1))

        self.reg_iou = nn.Sequential(*[CONV(self.internal_dim, self.internal_dim, 3, 1, 1, activation_function, self.batch_norm) for _ in range(num_convs_blocks)])

    def forward(self, X, scalar):
        '''
        Args:
            X (Tensor): FPN/FPNPAN output level
            scalar (float): reg output scalar
        
        Returns:
            results (Tensor): cls_output, reg_output, iou_output for neck output
        '''


        cls_output = self.cls(X)
        reg_iou = self.reg_iou(X)
        reg_output = self.relu(self.reg(reg_iou)*scalar)
        iou_output = self.iou(reg_iou)

        return (cls_output, reg_output, iou_output)


class ManyHeads(nn.Module):
    '''
    None anchor box object detector head - different head for each neck output

    Head generates 3 branches: IOU (objectness) 1xWxH, Reg 4xWxH (l,r,t,b), cls #clsxWxH

    Args:
        scalar (List[int]): list of parameters for each reg output level like FCOS
        internal_dim (int): number of channels inside Head
        class_number (int): number of classes
        activation (str): activation function
        batch_norm (bool): bool
        num_convs_blocks (int): number of CONV blocks
        
    '''
    def __init__(self, scalar, internal_dim, class_number, activation_function = "relu", batch_norm = False, num_convs_blocks = 4):
        super().__init__()

        self.Heads = nn.ModuleList([BaseHead(internal_dim=internal_dim, class_number=class_number, activation_function=activation_function, batch_norm=batch_norm, num_convs_blocks=num_convs_blocks) for _ in range(3)])
        self.scalar = scalar
    def forward(self, X):
        '''
        Args:
            X (List[Tensor]): FPN/FPNPAN output - [N3, N4, N5]
        
        Returns:
            results (List[Tensor]): list of cls_output, reg_output, iou_output for each level 
        '''

        output = []
        for idx, x in enumerate(X):

            output.append(self.Heads[idx](x, self.scalar[idx]))

        return output

class ManyHeadsExtended(nn.Module):
    '''
    None anchor box object detector head - different head for each neck output

    Head generates 3 branches: IOU (objectness) 1xWxH, Reg 4xWxH (l,r,t,b), cls #clsxWxH

    Args:
        scalar (List[int]): list of parameters for each reg output level like FCOS
        internal_dim (int): number of channels inside Head
        class_number (int): number of classes
        activation (str): activation function
        batch_norm (bool): bool
        num_convs_blocks (int): number of CONV blocks
        
    '''
    def __init__(self, scalar, internal_dim, class_number, activation_function = "relu", batch_norm = False, num_convs_blocks = 4):
        super().__init__()

        self.Heads = nn.ModuleList([BaseHead(internal_dim=internal_dim, class_number=class_number, activation_function=activation_function, batch_norm=batch_norm, num_convs_blocks=num_convs_blocks) for _ in range(4)])
        self.scalar = scalar
    def forward(self, X):
        '''
        Args:
            X (List[Tensor]): FPN/FPNPAN output - [N3, N4, N5]
        
        Returns:
            results (List[Tensor]): list of cls_output, reg_output, iou_output for each level 
        '''
        output = []
        for idx, x in enumerate(X):

            output.append(self.Heads[idx](x, self.scalar[idx]))

        return output

 