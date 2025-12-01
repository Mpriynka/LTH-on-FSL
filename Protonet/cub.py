import os

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class CUB(Dataset):
    def __init__(self, data_root, mode='test', transform=None):
        self.data_root = os.path.join(data_root, 'CUB_200_2011', 'CUB_200_2011')
        self.mode = mode
        self.transform = transform
        
        # Load metadata
        self.images_path = os.path.join(self.data_root, 'images')
        images_txt = os.path.join(self.data_root, 'images.txt')
        image_class_labels_txt = os.path.join(self.data_root, 'image_class_labels.txt')
        
        # Read image paths and labels
        # images.txt: <image_id> <image_name>
        # image_class_labels.txt: <image_id> <class_id>
        
        self.images = {}
        with open(images_txt, 'r') as f:
            for line in f:
                img_id, img_name = line.strip().split()
                self.images[int(img_id)] = img_name
                
        self.labels = {}
        with open(image_class_labels_txt, 'r') as f:
            for line in f:
                img_id, label = line.strip().split()
                self.labels[int(img_id)] = int(label)
                
        # Split: 1-100 Train, 101-150 Val, 151-200 Test
        if self.mode == 'train':
            self.selected_classes = range(1, 101)
        elif self.mode == 'val':
            self.selected_classes = range(101, 151)
        elif self.mode == 'test':
            self.selected_classes = range(151, 201)
        else:
            raise ValueError("Mode must be 'train', 'val', or 'test'")
            
        self.data = []
        self.targets = []
        
        for img_id, label in self.labels.items():
            if label in self.selected_classes:
                self.data.append(self.images[img_id])
                self.targets.append(label)
                
        self.label_to_indices = self._make_label_to_indices()

    def _make_label_to_indices(self):
        label_to_indices = {}
        for idx, label in enumerate(self.targets):
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)
        return label_to_indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data[idx]
        label = self.targets[idx]
        
        img_path = os.path.join(self.images_path, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
