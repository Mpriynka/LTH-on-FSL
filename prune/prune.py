import torch
import torch.nn as nn
import numpy as np

def get_masks(model, prune_rate):
    """
    Compute masks for the model based on magnitude pruning.
    prune_rate: percentage of weights to prune (e.g., 0.2 for 20% pruning)
    """
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = module.weight.data
            num_weights = weight.numel()
            num_prune = int(num_weights * prune_rate)
            
            if num_prune == 0:
                masks[name] = torch.ones_like(weight)
                continue
            
            # Flatten and sort
            flat_weight = torch.abs(weight.view(-1))
            threshold = torch.kthvalue(flat_weight, num_prune).values
            
            mask = torch.gt(torch.abs(weight), threshold).float()
            masks[name] = mask
            
    return masks

def apply_masks(model, masks):
    """
    Apply masks to the model weights (zero out pruned weights).
    Also ensures gradients for pruned weights are zeroed (optional, but good practice).
    """
    for name, module in model.named_modules():
        if name in masks:
            module.weight.data.mul_(masks[name])
            
def rewind_weights(model, w_init_state_dict, masks):
    """
    Reset model weights to w_init and apply masks.
    """
    model.load_state_dict(w_init_state_dict)
    apply_masks(model, masks)
    
def check_sparsity(model):
    total_zeros = 0
    total_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            weight = module.weight.data
            zeros = (weight == 0).sum().item()
            params = weight.numel()
            total_zeros += zeros
            total_params += params
            print(f"Layer {name}: {zeros}/{params} ({zeros/params:.2%}) pruned")
            
    print(f"Total sparsity: {total_zeros/total_params:.2%}")
