import torch
import numpy as np
import scipy.stats
from mini_imagenet import MiniImageNet, get_transforms
import torch.nn.functional as F

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, h

def meta_test(model, args, logger, mode='test'):
    model.eval()
    
    # Load Dataset
    # Use 'val' or 'test' transform and split
    transform = get_transforms(mode='test') # Val also uses test transform
    dataset = MiniImageNet(args.data_root, mode=mode, transform=transform)
    
    # Group by class
    # We need to access data directly to sample episodes efficiently
    # Let's build an index: class_idx -> [list of image indices]
    class_to_indices = {}
    for idx in range(len(dataset)):
        label = dataset.targets[idx]
        if label not in class_to_indices:
            class_to_indices[label] = []
        class_to_indices[label].append(idx)
        
    classes = list(class_to_indices.keys())
    
    accuracies = []
    
    with torch.no_grad():
        for episode in range(args.test_episodes):
            # 1. Sample 5 classes
            sampled_classes = np.random.choice(classes, args.n_way, replace=False)
            
            support_indices = []
            query_indices = []
            
            for cls in sampled_classes:
                # Sample 1 support + 15 query
                indices = class_to_indices[cls]
                # Ensure we have enough images
                if len(indices) < args.k_shot + args.k_query:
                     # Fallback or skip if not enough images (should not happen in standard miniImageNet)
                     continue
                     
                selected = np.random.choice(indices, args.k_shot + args.k_query, replace=False)
                support_indices.extend(selected[:args.k_shot])
                query_indices.extend(selected[args.k_shot:])
                
            # Load images
            support_imgs = []
            for idx in support_indices:
                img, _ = dataset[idx]
                support_imgs.append(img)
            support_imgs = torch.stack(support_imgs)
            
            query_imgs = []
            query_labels = [] # 0 to 4
            for i, idx in enumerate(query_indices):
                img, _ = dataset[idx]
                query_imgs.append(img)
                query_labels.append(i // args.k_query) # 0,0,..,1,1,..
            query_imgs = torch.stack(query_imgs)
            query_labels = torch.tensor(query_labels)
            
            if torch.cuda.is_available():
                support_imgs = support_imgs.cuda()
                query_imgs = query_imgs.cuda()
                query_labels = query_labels.cuda()
                
            # Extract Features
            support_feats = model(support_imgs, is_feat=True)
            query_feats = model(query_imgs, is_feat=True)
            
            # Normalize
            support_feats = F.normalize(support_feats, dim=1)
            query_feats = F.normalize(query_feats, dim=1)
            
            # Prototypes (Mean of support)
            support_feats = support_feats.view(args.n_way, args.k_shot, -1)
            prototypes = support_feats.mean(dim=1)
            
            # Nearest Neighbor
            dists = torch.cdist(query_feats, prototypes) # (k_query_total, n_way)
            
            pred = dists.argmin(dim=1)
            acc = (pred == query_labels).float().mean().item() * 100
            accuracies.append(acc)
            
            if (episode + 1) % 100 == 0:
                logger.info(f"Episode {episode+1}/{args.test_episodes}: Acc {acc:.2f}")
                
    mean, h = mean_confidence_interval(accuracies)
    logger.info(f"{mode.capitalize()} Acc: {mean:.2f} +/- {h:.2f}")
    return mean, h
