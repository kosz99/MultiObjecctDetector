import torch
import torch.nn as nn

class DIOULoss(nn.Module):
    '''
    DIOU Loss for regression bounding box compatible with FCOS fashion bounding box regression (l*, r*, t*, b*) - https://arxiv.org/pdf/1911.08287.pdf
    '''
    def __init__(self):
        super().__init__()
    
    def forward(self, input, label):
        '''
        Computing DIOU loss
        
        Args:
            input (Tensor): object detector bouding box regression map in FCOS fashion (batch_size x 4 x W x H)
            label (Tensor): object detector bouding box regression map label in FCOS fashion (batch_size x 4 x W x H)

        Results:
            return (Tensor): intersection over union loss (batch_size x 1 x W x H) 
    
        '''
        result = self.diou(input, label)


        return (1 - result).unsqueeze(1)
    
    def diou(self, input, label):
        '''
        Computing DIOU metric
        
        Args:
            input (Tensor): object detector bouding box regression map in FCOS fashion (batch_size x 4 x W x H)
            label (Tensor): object detector bouding box regression map label in FCOS fashion (batch_size x 4 x W x H)

        Results:
            return (Tensor): intersection over union metric (batch_size x W x H) 
    
        '''
        x1 = -input[:, 0, :, :]
        x2 = input[:,1,:,:]
        y1 = -input[:, 2, :, :]
        y2 = input[:, 3, :, :]

        label_x1 = -label[:, 0, :, :]
        label_x2 = label[:,1,:,:]
        label_y1 = -label[:, 2, :, :]
        label_y2 = label[:, 3, :, :]

        pred_center = ((x2-x1)/2,(y2-y1)/2)
        label_center = ((label_x2-label_x1)/2,(label_y2-label_y1)/2)

        center_distance_square = (pred_center[0]-label_center[0])**2 + (pred_center[1]-label_center[1])**2


        
        x1_intersection, _ = torch.max(torch.stack((x1, label_x1), dim = 1), dim = 1)
        x2_intersection, _ = torch.min(torch.stack((x2, label_x2), dim = 1), dim = 1)
        y1_intersection, _ = torch.max(torch.stack((y1, label_y1), dim = 1), dim = 1)
        y2_intersection, _ = torch.min(torch.stack((y2, label_y2), dim = 1), dim = 1)

        x1_convex, _ = torch.min(torch.stack((x1, label_x1), dim = 1), dim = 1)
        x2_convex, _ = torch.max(torch.stack((x2, label_x2), dim = 1), dim = 1)
        y1_convex, _ = torch.min(torch.stack((y1, label_y1), dim = 1), dim = 1)
        y2_convex, _ = torch.max(torch.stack((y2, label_y2), dim = 1), dim = 1)



        intersection = (x2_intersection + abs(x1_intersection))*((y2_intersection) + abs(y1_intersection))
        union = ((abs(x1)+x2)*(abs(y1)+y2)+(abs(label_x1)+label_x2)*(abs(label_y1)+label_y2)) - intersection
        convex_diagonal_length = (x2_convex-x1_convex)**2 + (y2_convex-y1_convex)**2

        IOU = intersection/(union+1e-8)

        return IOU - ((center_distance_square)/(convex_diagonal_length+1e-8))
    




    