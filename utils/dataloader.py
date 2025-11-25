import os
import os.path as osp
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch

class CIFARFS(Dataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        
        # Assuming standard CIFAR-FS split structure
        # data/cifar-fs/{train,val,test}
        # Inside each: class_name/image.png
        
        self.data_path = osp.join(root, mode)
        self.classes = sorted(os.listdir(self.data_path))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for c in self.classes:
            c_dir = osp.join(self.data_path, c)
            for img_name in os.listdir(c_dir):
                self.images.append(osp.join(c_dir, img_name))
                self.labels.append(self.class_to_idx[c])
                
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class MiniImageNet(Dataset):
    def __init__(self, root, mode='train', transform=None):
        self.root = root
        self.mode = mode
        self.transform = transform
        
        # Assuming standard miniImageNet csv structure or folder structure
        # Here implementing folder structure similar to CIFAR-FS for simplicity
        # data/mini-imagenet/{train,val,test}/class_name/image.jpg
        
        self.data_path = osp.join(root, mode)
        if not osp.exists(self.data_path):
             # Fallback to CSV if folders don't exist (common in miniImageNet)
             # But for now assuming pre-processed folder structure as user said "prepared directory structure"
             pass

        self.classes = sorted(os.listdir(self.data_path))
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        
        self.images = []
        self.labels = []
        
        for c in self.classes:
            c_dir = osp.join(self.data_path, c)
            for img_name in os.listdir(c_dir):
                self.images.append(osp.join(c_dir, img_name))
                self.labels.append(self.class_to_idx[c])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

class CategoriesSampler():
    def __init__(self, label, n_batch, n_cls, n_per):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            classes = torch.randperm(len(self.m_ind))[:self.n_cls]
            for c in classes:
                l = self.m_ind[c]
                pos = torch.randperm(len(l))[:self.n_per]
                batch.append(l[pos])
            batch = torch.stack(batch).t().reshape(-1)
            yield batch
