import os
import torch
from torch.utils.data import Dataset, Sampler
from torchvision import datasets, transforms
import numpy as np

class MiniImageNet(datasets.ImageFolder):
    def __init__(self, data_root, mode='train', transform=None):
        # Assuming data_root contains 'train', 'val', 'test' subdirectories
        root = os.path.join(data_root, mode)
        super(MiniImageNet, self).__init__(root, transform=transform)
        self.mode = mode
        self.label_to_indices = self._make_label_to_indices()

    def _make_label_to_indices(self):
        label_to_indices = {}
        for idx, (_, label) in enumerate(self.samples):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

class CategoriesSampler(Sampler):
    def __init__(self, label_to_indices, n_batch, n_cls, n_per):
        self.n_batch = n_batch # number of episodes per epoch
        self.n_cls = n_cls # n_way
        self.n_per = n_per # k_shot + k_query
        self.label_to_indices = label_to_indices
        self.classes = list(self.label_to_indices.keys())

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.classes))[:self.n_cls]
            for c in classes:
                l = self.classes[c]
                indices = self.label_to_indices[l]
                pos = torch.randperm(len(indices))[:self.n_per]
                batch.append(torch.tensor(indices)[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch

def get_transforms(mode='train'):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    if mode == 'train':
        return transforms.Compose([
            transforms.Resize(84),
            transforms.CenterCrop(84), 
            transforms.RandomResizedCrop(84),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        return transforms.Compose([
            transforms.Resize((84, 84)), # Resize to 84x84 directly
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
