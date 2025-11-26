import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

def euclidean_dist(x, y):
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)

def protonet_loss(model, data, n_way, k_shot, q_query, device):
    # data: (n_way * (k_shot + q_query), C, H, W)
    # Support: n_way * k_shot
    # Query: n_way * q_query
    
    embeddings = model(data)
    z_dim = embeddings.size(1)
    
    # Reshape to (n_way, k_shot + q_query, z_dim)
    # Assuming data is sorted by class
    embeddings = embeddings.reshape(n_way, k_shot + q_query, z_dim)
    
    support = embeddings[:, :k_shot] # (n_way, k_shot, z_dim)
    query = embeddings[:, k_shot:]   # (n_way, q_query, z_dim)
    
    # Prototypes: (n_way, z_dim)
    prototypes = support.mean(1)
    
    # Query: (n_way * q_query, z_dim)
    query = query.contiguous().view(n_way * q_query, z_dim)
    
    # Distance: (n_way * q_query, n_way)
    dists = euclidean_dist(query, prototypes)
    
    # Log Softmax
    log_p_y = F.log_softmax(-dists, dim=1)
    
    # Targets
    target_inds = torch.arange(0, n_way).view(n_way, 1, 1).expand(n_way, q_query, 1).long()
    target_inds = target_inds.to(device)
    target_inds = target_inds.reshape(-1)
    
    loss = F.nll_loss(log_p_y, target_inds)
    
    # Accuracy
    _, y_hat = log_p_y.max(1)
    acc = torch.eq(y_hat, target_inds).float().mean()
    
    return loss, acc

def train_protonet(model, train_loader, optimizer, epoch, n_way, k_shot, q_query, device):
    model.train()
    total_loss = 0
    total_acc = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    for batch_idx, (data, _) in enumerate(pbar):
        # data comes as (batch_size, C, H, W) where batch_size = n_way * (k_shot + q_query)
        data = data.to(device)
        
        optimizer.zero_grad()
        loss, acc = protonet_loss(model, data, n_way, k_shot, q_query, device)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_acc += acc.item()
        pbar.set_postfix(loss=loss.item(), acc=acc.item())
        
    return total_loss / len(train_loader), total_acc / len(train_loader)

def eval_protonet(model, test_loader, n_way, k_shot, q_query, device):
    model.eval()
    accs = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            _, acc = protonet_loss(model, data, n_way, k_shot, q_query, device)
            accs.append(acc.item())
            
    accs = torch.tensor(accs)
    mean = accs.mean().item() * 100
    std = accs.std().item() * 100
    ci = 1.96 * std / (len(accs) ** 0.5)
            
    return mean, ci
