import torch
import torch.nn as nn
import torchvision.ops as ops

class MulticlassBCELoss(nn.Module):
    '''
    Loss for multiclass binary classification
    '''
    def __init__(self):
        super().__init__()
        #self.sigmoid = nn.Sigmoid()
        self.bceloss = nn.BCEWithLogitsLoss(reduction="none")
    
    def forward(self, input, label, mask):
        '''
        Computing Multiclass BCELoss
        
        Args:
            input (Tensor): Classification head output (batch_size x #cls x W x H)
            label (Tensor): Zero one hot encoding classification label (batch_size x #cls x W x H)
            mask (Tensor): mask indicates positive positions (batch_size x 1 x W x H)

        Results:
            return (Tensor): Classification loss 
    
        '''
        #input = self.sigmoid(input)
        loss = torch.zeros_like(input)

        for i in range(input.shape[1]):
            loss[:,i,:,:] = self.bceloss(input[:,i,:,:], label[:,i,:,:])
        loss = loss.sum(dim=1).unsqueeze(1)
        loss = loss/input.shape[1]
        loss = loss * mask
        loss = loss.sum()

        return loss


class MulticlassFocalLoss(nn.Module):
    '''
    Loss for multiclass binary classification (Focal)
    '''
    def __init__(self):
        super().__init__()

    
    def forward(self, input, label, mask):
        '''
        Computing Multiclass BCELoss
        
        Args:
            input (Tensor): Classification head output (batch_size x #cls x W x H)
            label (Tensor): Zero one hot encoding classification label (batch_size x #cls x W x H)
            mask (Tensor): mask indicates positive positions (batch_size x 1 x W x H)

        Results:
            return (Tensor): Classification loss 
    
        '''
       
        loss = torch.zeros_like(input)

        for i in range(input.shape[1]):
            loss[:,i,:,:] = ops.sigmoid_focal_loss(input[:,i,:,:], label[:,i,:,:], reduction="none")
        loss = loss.sum(dim=1).unsqueeze(1)
        loss = loss/input.shape[1]
        loss = loss * mask
        loss = loss.sum()

        return loss

