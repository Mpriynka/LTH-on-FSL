import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock(nn.Module):
    def __init__(self, block_size):
        super(DropBlock, self).__init__()
        self.block_size = block_size

    def forward(self, x, gamma):
        if self.training:
            batch_size, channels, height, width = x.shape
            
            bernoulli = torch.distributions.Bernoulli(gamma)
            mask = bernoulli.sample((batch_size, channels, height - (self.block_size - 1), width - (self.block_size - 1))).cuda()
            block_mask = self._compute_block_mask(mask)
            countM = block_mask.size()[0] * block_mask.size()[1] * block_mask.size()[2] * block_mask.size()[3]
            count_ones = block_mask.sum()

            return block_mask * x * (countM / count_ones)
        else:
            return x

    def _compute_block_mask(self, mask):
        left_padding = int((self.block_size - 1) / 2)
        right_padding = int(self.block_size / 2)
        
        batch_size, channels, height, width = mask.shape
        non_zero_idxs = mask.nonzero()
        nr_blocks = non_zero_idxs.shape[0]

        offsets = torch.stack(
            [
                torch.arange(self.block_size).view(-1, 1).expand(self.block_size, self.block_size).reshape(-1),
                torch.arange(self.block_size).view(1, -1).expand(self.block_size, self.block_size).reshape(-1)
            ]
        ).t().cuda()
        offsets = torch.cat((torch.zeros(self.block_size**2, 2).cuda().long(), offsets.long()), 1)
        
        if nr_blocks > 0:
            non_zero_idxs = non_zero_idxs.repeat(self.block_size ** 2, 1)
            offsets = offsets.repeat(nr_blocks, 1).view(-1, 4)
            offsets = offsets.long()

            block_idxs = non_zero_idxs + offsets
            #padded_mask = F.pad(mask, (left_padding, right_padding, left_padding, right_padding))
            padded_mask = torch.zeros((batch_size, channels, height + self.block_size - 1, width + self.block_size - 1)).cuda()
            padded_mask[block_idxs[:, 0], block_idxs[:, 1], block_idxs[:, 2], block_idxs[:, 3]] = 1.
        else:
            padded_mask = torch.zeros((batch_size, channels, height + self.block_size - 1, width + self.block_size - 1)).cuda()
            
        return 1 - padded_mask[:, :, left_padding:-right_padding, left_padding:-right_padding]

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes)
            )
            
        self.maxpool = nn.MaxPool2d(stride)
        self.drop_rate = drop_rate
        self.drop_block = drop_block
        self.block_size = block_size
        self.DropBlock = DropBlock(block_size=self.block_size)

    def forward(self, x):
        out = F.leaky_relu(self.bn1(self.conv1(x)), negative_slope=0.1)
        out = F.leaky_relu(self.bn2(self.conv2(out)), negative_slope=0.1)
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.leaky_relu(out, negative_slope=0.1)
        out = self.maxpool(out)
        
        if self.drop_rate > 0:
            if self.drop_block:
                feat_size = out.size()[2]
                keep_rate = max(1.0 - self.drop_rate / (20*2000) * (self.num_batches_tracked), 1.0 - self.drop_rate)
                gamma = (1 - keep_rate) / self.block_size**2 * feat_size**2 / (feat_size - self.block_size + 1)**2
                out = self.DropBlock(out, gamma=gamma)
            else:
                out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
                
        return out

class ResNet12(nn.Module):
    def __init__(self, feature_dim=640, drop_rate=0.0, drop_block=False, block_size=1):
        super(ResNet12, self).__init__()
        self.in_planes = 3
        self.layer1 = self._make_layer(BasicBlock, 64, stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=block_size)
        self.layer2 = self._make_layer(BasicBlock, 160, stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=block_size)
        self.layer3 = self._make_layer(BasicBlock, 320, stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=block_size)
        self.layer4 = self._make_layer(BasicBlock, 640, stride=2, drop_rate=drop_rate, drop_block=drop_block, block_size=block_size)
        
        self.keep_avg_pool = True
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(640, feature_dim)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        layers = []
        layers.append(block(self.in_planes, planes, stride, drop_rate, drop_block, block_size))
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.keep_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
