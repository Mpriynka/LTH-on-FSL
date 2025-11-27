import torch
import torch.nn as nn

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class Conv4(nn.Module):
    def __init__(self, num_classes=64):
        super(Conv4, self).__init__()
        self.encoder = nn.Sequential(
            conv_block(3, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            conv_block(64, 64)
        )
        self.feat_dim = 64 * 5 * 5 # 1600
        self.fc = nn.Linear(self.feat_dim, num_classes)

    def forward(self, x, is_feat=False):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        
        if is_feat:
            return x
            
        x = self.fc(x)
        return x

def conv4(**kwargs):
    return Conv4(**kwargs)
