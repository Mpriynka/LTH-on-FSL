import torch
import torch.nn as nn

class Conv4(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(Conv4, self).__init__()
        self.encoder = nn.Sequential(
            self.conv_block(x_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, hid_dim),
            self.conv_block(hid_dim, z_dim),
        )

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.size(0), -1)
