import torch
import torch.nn as nn
import numpy as np

def get_masks(model, prune_ratio):
    """
    Calculate masks for global magnitude pruning of Conv2d weights.
    prune_ratio: float, percentage of weights to prune (e.g., 90 for 90% sparsity).
    """
    # Gather all weights from Conv2d layers
    all_weights = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            all_weights.append(module.weight.data.abs().view(-1))
    
    if not all_weights:
        return {}

    all_weights = torch.cat(all_weights)
    
    # Calculate threshold
    k = int(len(all_weights) * prune_ratio / 100)
    if k == 0:
        return {}
        
    threshold, _ = torch.topk(all_weights, k, largest=False)
    threshold = threshold[-1]
    
    # Generate masks
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            mask = (module.weight.data.abs() > threshold).float()
            masks[name] = mask
            
    return masks

def apply_masks(model, masks):
    """
    Applies masks to model weights (zeros out pruned weights).
    """
    for name, module in model.named_modules():
        if name in masks:
            module.weight.data.mul_(masks[name])

def apply_mask_grads(model, masks):
    """
    Zeros out gradients for pruned weights.
    Should be called after backward().
    """
    for name, module in model.named_modules():
        if name in masks and module.weight.grad is not None:
            module.weight.grad.data.mul_(masks[name])

def current_sparsity(model):
    """
    Calculates global sparsity of Conv2d layers.
    """
    total_params = 0
    zero_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            params = module.weight.data.numel()
            zeros = (module.weight.data == 0).sum().item()
            total_params += params
            zero_params += zeros
    
    if total_params == 0: return 0
    return 100.0 * zero_params / total_params
