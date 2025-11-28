import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock(nn.Module):
    def __init__(self, block_size=5, keep_prob=0.9):
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        gamma = (1. - self.keep_prob) / self.block_size ** 2
        for sh in x.shape[2:]:
            gamma *= sh / (sh - self.block_size + 1)
        mask = torch.bernoulli(torch.ones_like(x) * gamma)
        mask_block = 1 - F.max_pool2d(mask,
                                      kernel_size=(self.block_size, self.block_size),
                                      stride=(1, 1),
                                      padding=(self.block_size // 2, self.block_size // 2))
        x = mask_block * x * (mask_block.numel() / mask_block.sum())
        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size, keep_prob=1-self.drop_rate)

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block == True:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                self.DropBlock.keep_prob = keep_rate
                self.DropBlock.block_size = 5
                out = self.DropBlock(out)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)

        return out

class ResNet12(nn.Module):
    def __init__(self, avg_pool=True, drop_rate=0.1, drop_block=False, num_classes=64):
        super(ResNet12, self).__init__()
        self.inplanes = 3
        self.layer1 = self._make_layer(BasicBlock, 64, stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=5)
        self.layer2 = self._make_layer(BasicBlock, 160, stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=5)
        self.layer3 = self._make_layer(BasicBlock, 320, stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=5)
        self.layer4 = self._make_layer(BasicBlock, 640, stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=5)
        self.avg_pool = avg_pool
        self.keep_avg_pool = avg_pool
        self.feat_dim = 640
        
        # Classifier
        self.fc = nn.Linear(640, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes
        return nn.Sequential(*layers)

    def forward(self, x, is_feat=False):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        if self.avg_pool:
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            
        if is_feat:
            return x
            
        x = self.fc(x)
        return x

def resnet12(**kwargs):
    """Constructs a ResNet-12 model."""
    return ResNet12(**kwargs)
