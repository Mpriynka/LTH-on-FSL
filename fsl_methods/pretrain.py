import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def train_pretrain(model, classifier, train_loader, optimizer, epoch, device):
    model.train()
    classifier.train()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        features = model(data)
        output = classifier(features)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix(loss=loss.item())
        
    return total_loss / len(train_loader)
