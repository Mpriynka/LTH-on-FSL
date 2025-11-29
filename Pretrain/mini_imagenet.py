import os
import torch
from torchvision import datasets, transforms

class MiniImageNet(datasets.ImageFolder):
    def __init__(self, data_root, mode='train', transform=None):
        root = os.path.join(data_root, mode)
        super(MiniImageNet, self).__init__(root, transform=transform)
        self.mode = mode

def get_transforms(mode='train'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            transforms.RandomResizedCrop(84), 
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
