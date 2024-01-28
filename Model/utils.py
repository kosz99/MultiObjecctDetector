import torch
import torch.nn as nn

from Loss.IOULoss import IOULoss
from Loss.GIOULoss import GIOULoss
from Loss.DIOULoss import DIOULoss
from Loss.FocalLoss import FocalLoss
from Loss.multiclassBCELoss import MulticlassBCELoss, MulticlassFocalLoss 

def get_lrScheduler(name, optimizer, max_lr, steps_per_epoch, epochs, T_max, eta_min, verbose = True):
    assert name == "OneCycle" or name == "Cosine", f"Wrong scheduler name"
    func = {
        "OneCycle" : torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, steps_per_epoch=steps_per_epoch, verbose=verbose),
        "Cosine" : torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min, verbose=verbose)
    }
    return func[name]
def get_reg_loss(loss_name):
    assert loss_name == "IOU" or loss_name == "GIOU" or loss_name == "DIOU", f"Reg loss name error"
    func = {
        "IOU" : IOULoss(),
        "GIOU" : GIOULoss(),
        "DIOU" : DIOULoss()
    }
    return func[loss_name]

def get_objectness_loss(loss_name):
    assert loss_name == "Focal", f"Objectness loss error"
    func = {
        "Focal" : FocalLoss()
    }
    return func[loss_name]

def get_cls_loss(loss_name):
    assert loss_name == "BCE" or loss_name == "Focal", f"Cls loss error"
    func = {
        "Focal" : MulticlassFocalLoss(),
        "BCE" : MulticlassBCELoss()
    }
    return func[loss_name]

def get_optimizer(model, optimizer_name, learning_rate, momentum = 0, weight_decay = 0):
    assert optimizer_name == "SGD" or optimizer_name == "ADAM" or optimizer_name == "ADAMW", f"unknown optimzer"
    optim = {
        "SGD" : torch.optim.SGD(model.parameters(), learning_rate, momentum=momentum, weight_decay=weight_decay),
        "ADAM" : torch.optim.Adam(model.parameters(), learning_rate, weight_decay=weight_decay),
        "ADAMW" : torch.optim.AdamW(model.parameters(), learning_rate, weight_decay=weight_decay)
    }
    return optim[optimizer_name]

def get_activation(activation_name: str, negative_slope: int = 0.01):
    func = {
        "relu": nn.ReLU(),
        "elu": nn.ELU(),
        "celu": nn.CELU(),
        "mish": nn.Mish(),
        "gelu": nn.GELU(),
        "leaky_relu": nn.LeakyReLU(negative_slope=negative_slope),
        "silu": nn.SiLU()
    }
    return func[activation_name]

def get_upsample(upsample_type = "nearest", scale_factor = 2):
    assert upsample_type == "nearest" or upsample_type == "bilinear", f"Wrong upsample type"
    func = {
        "nearest": nn.UpsamplingNearest2d(scale_factor = scale_factor),
        "bilinear": nn.UpsamplingBilinear2d(scale_factor = scale_factor)
    }

    return func[upsample_type]
class CONV(nn.Module):
    '''
    CONV module - Conv2d + batch norm(optional) + activation function

    Args:
        input_channels (int): number of nn.Conv2d input channels
        output_channels (int): number of nn.Conv2d output_channels
        kernel_size (int): nn.Conv2d kernel size
        stride (int): nn.Conv2d stride
        padding (int): nn.Conv2d padding
        act (String): activation function
        batch_norm (bool): batch norm 
    '''
    def __init__(self, input_channels, output_channels, kernel_size, stride = 1, padding = 0,  act = "relu", batch_norm = False):
        super().__init__()

        if batch_norm:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(output_channels),
                get_activation(act)

            )
        
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride, padding, bias = True),
                get_activation(act)

            )
    
    def forward(self, x):
        '''
        Module computing

        Args:
            input (Tensor): module input
        
        Returns:
            results (Tensor): module output (nn.Sequential - conv2d, bn, act)
        '''

        return self.conv(x)

class DWCONV(nn.Module):
    '''
    DWCONV(Depthwise Convolution) module - Conv2d + batch norm(optional) + activation function

    Args:
        channels (int): number of nn.Conv2d input/output channels
        kernel_size (int): nn.Conv2d kernel size
        stride (int): nn.Conv2d stride
        padding (int): nn.Conv2d padding
        act (String): activation function
        batch_norm (bool): batch norm 
    '''
    def __init__(self, channels, kernel_size, stride = 1, padding = 0,  act = "relu", batch_norm = False):
        super().__init__()

        if batch_norm:
            self.dwconv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, stride, padding, groups = channels, bias = False),
                nn.BatchNorm2d(channels),
                get_activation(act)

            )
        
        else:
            self.dwconv = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size, stride, padding, groups = channels, bias = True),
                get_activation(act)

            )
    
    def forward(self, x):
        '''
        Module computing

        Args:
            input (Tensor): module input
        
        Returns:
            results (Tensor): module output (nn.Sequential - dwconv2d, bn, act)
        '''

        return self.dwconv(x)
    
class GSCONV(nn.Module):
    '''
    Add comment ;)
    '''
    def __init__(self, input_channels, output_channels, kernel_size, stride, padding, act = "relu", batch_norm = True):
        super().__init__()
        self.conv = CONV(input_channels, output_channels//2, kernel_size, stride, padding, act, batch_norm)
        self.dw = DWCONV(output_channels//2, 3, 1, 1, act, batch_norm)

    def forward(self, x):
        x = self.conv(x)
        dw_out = self.dw(x)
        output = torch.cat((x, dw_out), dim=1)

        return output

class GSBottleneck(nn.Module):
    def __init__(self, channels_num, act = 'relu', batch_norm = False):
        super().__init__()
        self.gsconv = nn.Sequential(
            GSCONV(channels_num, channels_num//2, 3, 1, 1, act, batch_norm),
            GSCONV(channels_num//2, channels_num, 3, 1, 1, act, batch_norm)
        )
        self.shortcut_conv = CONV(channels_num, channels_num, 1, 1, 0, act, batch_norm)
    
    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        x = self.gsconv(x)
        
        return x + shortcut

class VoVGSCSP(nn.Module):
    def __init__(self, input_channels_num, channels_num, output_channels_num, act = 'relu', batch_norm = False, bottleneck_num = 1):
        super().__init__()
        self.bottleneck = nn.Sequential(*[GSBottleneck(channels_num, act, batch_norm) for _ in range(bottleneck_num)])
        self.input_conv = CONV(input_channels_num, channels_num, 1, 1, 0, act, batch_norm)
        self.shortcut_conv = CONV(input_channels_num, channels_num, 1, 1, 0, act, batch_norm)
        self.output_conv = CONV(channels_num*2, output_channels_num, 1, 1, 0, act, batch_norm)
    
    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        x = self.input_conv(x)
        x = self.bottleneck(x)
        x = torch.cat((x, shortcut), dim = 1)
        x = self.output_conv(x)

        return x        

        
class BottleNeck(nn.Module):
    def __init__(self, channels_num, act = 'relu', batch_norm = False, shortcut = False):
        super().__init__()

        self.shortcut = shortcut
        self.conv = nn.Sequential(
            CONV(channels_num, channels_num, kernel_size=1, stride=1, padding=0, act=act, batch_norm=batch_norm),
            CONV(channels_num, channels_num, kernel_size=3, stride=1, padding=1, act=act, batch_norm=batch_norm)
        )

    def forward(self, x):
        if self.shortcut == True:
            return x + self.conv(x)
        else:
            return self.conv(x)

class C3_Module(nn.Module):
    def __init__(self, input_channels_num, channels_num, output_channels_num, act = 'relu', batch_norm = False, bottleneck_shortcut = False, bottleneck_num = 3):
        super().__init__()

        if bottleneck_shortcut:
            self.bottleneck = nn.Sequential(*[BottleNeck(channels_num, act=act, batch_norm=batch_norm, shortcut=bottleneck_shortcut) for _ in range(bottleneck_num)])
        else:
            self.bottleneck = nn.Sequential(*[BottleNeck(channels_num, act=act, batch_norm=batch_norm, shortcut=False) for _ in range(bottleneck_num)])
        
        self.input_conv = CONV(input_channels_num, channels_num, kernel_size=1, stride=1, padding=0, act=act, batch_norm=batch_norm)
        self.shortcut_conv = CONV(input_channels_num, channels_num, kernel_size=1, stride=1, padding=0, act=act, batch_norm=batch_norm)
        self.output_conv = CONV(channels_num*2, output_channels_num, kernel_size=1, stride=1, padding=0, act=act, batch_norm=batch_norm)

    def forward(self, x):
        shortcut = self.shortcut_conv(x)
        x = self.input_conv(x)
        x = self.bottleneck(x)
        x = torch.cat((x, shortcut), dim = 1)
        x = self.output_conv(x)

        return x


