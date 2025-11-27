import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy

from data.mini_imagenet import MiniImageNet, get_transforms
from models.resnet12 import resnet12
from utils.utils import set_seed, get_logger, save_checkpoint
from utils.pruning import prune_model, apply_mask, current_sparsity
from train import train_epoch
from evaluate import meta_test

def get_args():
    parser = argparse.ArgumentParser(description='LTH for FSL (Pretrain)')
    parser.add_argument('--data_root', type=str, default='./data/miniImagenet', help='path to dataset')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='path to save checkpoints')
    parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--gpu', type=str, default='0', help='gpu device id')
    parser.add_argument('--print_freq', type=int, default=300, help='print frequency')
    
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--n_shot', type=int, default=1)
    parser.add_argument('--n_query', type=int, default=15)
    parser.add_argument('--test_episodes', type=int, default=2000)
    
    parser.add_argument('--prune_ratios', type=float, nargs='+', default=[10, 50, 90], help='Pruning ratios (LS, MS, HS)')
    
    return parser.parse_args()

def main():
    args = get_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        
    logger = get_logger(os.path.join(args.save_dir, 'train.log'))
    logger.info(args)
    set_seed(args.seed)
    
    # Data Loaders
    train_transform = get_transforms('train')
    val_transform = get_transforms('val') 
    
    train_set = MiniImageNet(args.data_root, 'train', train_transform)
    val_set = MiniImageNet(args.data_root, 'val', val_transform)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    
    # Model
    model = resnet12(num_classes=64) 
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Save Init Weights
    W_init = copy.deepcopy(model.state_dict())
    torch.save(W_init, os.path.join(args.save_dir, 'W_init.pth'))
    
    # --- Step 1: Train Dense Network ---
    logger.info("Starting Dense Training...")
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1) # Standard decay
    
    best_acc = 0
    dense_path = os.path.join(args.save_dir, 'model_dense_best.pth')
    
    # Check if already trained
    if os.path.exists(dense_path):
        logger.info(f"Loading existing dense model from {dense_path}")
        checkpoint = torch.load(dense_path)
        model.load_state_dict(checkpoint['state_dict'])
    else:
        for epoch in range(args.epochs):
            train_acc, train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, args, logger)
            
            # Episodic Validation
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                val_acc, val_ci = meta_test(model, args, logger, mode='val')
                model.train() # Switch back to train mode
            else:
                val_acc, val_ci = 0, 0

            scheduler.step()
            
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
            }, is_best, filename='checkpoint_dense.pth', folder=args.save_dir)
            
            if is_best:
                torch.save({'state_dict': model.state_dict()}, dense_path)
                
    # Evaluate Dense
    logger.info("Evaluating Dense Model...")
    # Load best dense
    checkpoint = torch.load(dense_path)
    model.load_state_dict(checkpoint['state_dict'])
    dense_acc, dense_ci = meta_test(model, args, logger)
    logger.info(f"Dense Result: {dense_acc:.2f} +/- {dense_ci:.2f}")
    
    W_dense = copy.deepcopy(model.state_dict())
    
    # --- Step 2 & 3: Prune and Retrain ---
    for ratio in args.prune_ratios:
        logger.info(f"--- Pruning Ratio: {ratio}% ---")
        
        # 1. Load Dense Weights to calculate mask
        model.load_state_dict(W_dense)
        
        # 2. Generate Mask
        masks = prune_model(model, ratio)
        logger.info(f"Sparsity: {current_sparsity(model):.2f}%")
        
        # 3. Reset to W_init
        model.load_state_dict(W_init)
        
        # 4. Apply Mask (to weights initially)
        apply_mask(model, masks)
        
        # 5. Retrain
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 80], gamma=0.1)
        
        best_subnet_acc = 0
        subnet_path = os.path.join(args.save_dir, f'model_subnet_{ratio}.pth')
        
        for epoch in range(args.epochs):
            # Pass masks to zero out gradients
            train_acc, train_loss = train_epoch(train_loader, model, criterion, optimizer, epoch, args, logger, masks=masks)
            
            # Apply mask to weights again just in case (though grads are masked)
            apply_mask(model, masks)
            
            # Episodic Validation
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                val_acc, val_ci = meta_test(model, args, logger, mode='val')
                model.train() # Switch back to train mode
            else:
                val_acc, val_ci = 0, 0


            scheduler.step()
            
            is_best = val_acc > best_subnet_acc
            best_subnet_acc = max(val_acc, best_subnet_acc)
            
            if is_best:
                torch.save({'state_dict': model.state_dict(), 'masks': masks}, subnet_path)
                
        # Evaluate Subnet
        logger.info(f"Evaluating Subnet {ratio}%...")
        checkpoint = torch.load(subnet_path)
        model.load_state_dict(checkpoint['state_dict'])
        apply_mask(model, masks) # Ensure masked
        
        subnet_acc, subnet_ci = meta_test(model, args, logger)
        logger.info(f"Subnet {ratio}% Result: {subnet_acc:.2f} +/- {subnet_ci:.2f}")

if __name__ == '__main__':
    main()
