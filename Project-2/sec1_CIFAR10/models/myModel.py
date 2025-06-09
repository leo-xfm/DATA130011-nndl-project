import torch
import torch.nn as nn
import timm
from models.dynamic_tanh import convert_ln_to_dyt

class convBlock(nn.Module):
    def __init__(self, in_dim, out_dim, res=True):
        super(convBlock, self).__init__()
        self.res = res
        self.conv1 = nn.Sequential(
            # nn.Conv2d(in_dim, out_dim, kernel_size=5, padding=2), # [B, C, in, out]
            nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), # [B, C, in, out]
            nn.BatchNorm2d(out_dim),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=0.01)
            # nn.ELU(alpha=1.0)
            # nn.GELU()
        )
        self.conv2 = nn.Sequential(
            # nn.Conv2d(out_dim, out_dim, kernel_size=5, padding=2),
            nn.Conv2d(out_dim, out_dim, kernel_size=3, padding=1), # [B, C, in, out]
            nn.BatchNorm2d(out_dim),
        )
    
    def forward(self, x):
        out = self.conv1(x) 
        res = out if self.res else 0
        out = self.conv2(out)
        return out + res

class myModel(nn.Module):
    def __init__(self, num_classes=10,
                 cfg=[64, 128, 256, 512],  # [64, 128, 256, 512]
                 res=True, dropout=0.4):
        super(myModel, self).__init__()
        
        self.num_classes = num_classes
        self.cfg = cfg
        self.in_dim = 3
        self.res = res
        self.dropout = dropout
        
        self.convBlock = self.constructConv()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4*cfg[-1], num_classes)
        
        self.vit_branch = timm.create_model('vit_tiny_r_s16_p8_384', pretrained=True, num_classes=10)
        self.vit_branch = convert_ln_to_dyt(self.vit_branch)
    
    def constructConv(self):
        layers = []
        in_dim = self.in_dim
        for neuron_num in self.cfg:
            layers.append(convBlock(in_dim, neuron_num, self.res))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_dim = neuron_num
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # [bsz, 3, 32, 32]
        out1 = self.convBlock(x)
        out1 = self.flatten(out1)
        out1 = self.dropout(out1)
        out1 = self.fc(out1)
        
        vit_input = nn.functional.interpolate(x, size=(384, 384), mode='bilinear', align_corners=False)
        out2 = self.vit_branch(vit_input)
        # print(out2.shape)
        
        # return out1
        # return out2
        return out1 + out2


class myCNNModel(nn.Module):
    def __init__(self, num_classes=10,
                 cfg=[64, 128, 256, 512],  # [64, 128, 256, 512]
                 res=True, dropout=0.4):
        super(myCNNModel, self).__init__()
        
        self.num_classes = num_classes
        self.cfg = cfg
        self.in_dim = 3
        self.res = res
        self.dropout = dropout
        
        self.convBlock = self.constructConv()
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(4*cfg[-1], num_classes) # change!
    
    def constructConv(self):
        layers = []
        in_dim = self.in_dim
        for neuron_num in self.cfg:
            layers.append(convBlock(in_dim, neuron_num, self.res))
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_dim = neuron_num
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # [bsz, 3, 32, 32]
        out1 = self.convBlock(x)
        out1 = self.flatten(out1)
        out1 = self.dropout(out1)
        out1 = self.fc(out1)
        return out1