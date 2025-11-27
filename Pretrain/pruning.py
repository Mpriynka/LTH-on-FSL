import torch
import torch.nn as nn
import numpy as np

def prune_model(model, percent):
    """
    Generates masks for Conv2d layers to prune 'percent' of weights (layer-wise).
    percent: 0 to 100
    Returns: dict of masks {layer_name: binary_mask}
    """
    masks = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            weight = module.weight.data.abs().cpu().numpy()
            percentile = np.percentile(weight, percent)
            mask = torch.from_numpy((weight >= percentile).astype(float)).float().to(module.weight.device)
            masks[name] = mask
    return masks

def apply_mask(model, masks):
    """
    Applies masks to model weights (zeros out pruned weights).
    Also sets gradients of pruned weights to 0 (using hooks or manual zeroing).
    Here we just zero weights. Gradient masking should be done in training loop.
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
