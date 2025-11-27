import torch
import torch.nn as nn
import torch.nn.functional as F

class ProtoNet(nn.Module):
    def __init__(self, backbone):
        super(ProtoNet, self).__init__()
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x, is_feat=True)

    def proto_loss(self, x, n_way, k_shot, k_query):
        """
        x: Input images of shape (n_way * (k_shot + k_query), C, H, W)
        """
        # Extract features
        z = self.backbone(x, is_feat=True) # (batch_size, feat_dim)
        
        # Reshape to (n_way, k_shot + k_query, feat_dim)
        z = z.view(n_way, k_shot + k_query, -1)
        
        # Split into support and query
        support = z[:, :k_shot] # (n_way, k_shot, feat_dim)
        query = z[:, k_shot:]   # (n_way, k_query, feat_dim)
        
        # Calculate prototypes
        prototypes = support.mean(dim=1) # (n_way, feat_dim)
        
        # Calculate distances (Euclidean)
        # query: (n_way, k_query, feat_dim) -> (n_way * k_query, feat_dim)
        query = query.contiguous().view(n_way * k_query, -1)
        
        # prototypes: (n_way, feat_dim)
        # dists: (n_way * k_query, n_way)
        dists = torch.cdist(query, prototypes)
        
        # Calculate Logits (negative distance)
        logits = -dists
        
        # Create labels
        # The query samples are ordered by class: 
        # class 0 (k_query samples), class 1 (k_query samples), ...
        labels = torch.arange(n_way).unsqueeze(1).repeat(1, k_query).view(-1)
        if x.is_cuda:
            labels = labels.cuda()
            
        # Calculate Loss and Accuracy
        loss = F.cross_entropy(logits, labels)
        
        # Accuracy
        _, pred = logits.max(1)
        acc = pred.eq(labels).float().mean()
        
        return loss, acc * 100
