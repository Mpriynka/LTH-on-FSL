import argparse
import os
import time
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from mini_imagenet import MiniImageNet, CategoriesSampler, get_transforms
from backbone.conv4 import conv4
from backbone.resnet12 import resnet12
from protonet import ProtoNet
from utils import set_seed, AverageMeter, save_checkpoint, get_logger, mean_confidence_interval
from pruning import get_masks, apply_masks, apply_mask_grads, current_sparsity

def parse_args():
    parser = argparse.ArgumentParser(description='LTH for FSL (ProtoNet)')
    parser.add_argument('--data-root', type=str, default='Datasets/Mini-Imagenet', help='Path to dataset')
    parser.add_argument('--backbone', type=str, default='conv4', choices=['conv4', 'resnet12'], help='Backbone model')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of support samples per class')
    parser.add_argument('--k_query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes per epoch') 
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default='./checkpoints/Protonet', help='Output directory')
    parser.add_argument('--print-freq', type=int, default=600, help='Print frequency')
    parser.add_argument('--meta-batch-size', type=int, default=1, help='Meta-batch size (number of episodes to average over)')
    parser.add_argument('--num-workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--test-episodes', type=int, default=2000, help='Number of episodes for testing')
    return parser.parse_args()

def train_epoch(model, loader, optimizer, epoch, args, logger, masks=None):
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accs = AverageMeter()
    
    end = time.time()
    
    optimizer.zero_grad()
    
    for i, (data, _) in enumerate(loader):
        data_time.update(time.time() - end)
        
        if torch.cuda.is_available():
            data = data.cuda()
            
        # Reshape data for ProtoNet: (n_way * (k_shot + k_query), C, H, W)
        # The sampler yields a batch of indices which the dataset converts to images.
        # The batch size is n_way * (k_shot + k_query).
        
        loss, acc = model.proto_loss(data, args.n_way, args.k_shot, args.k_query)
        
        # Normalize loss by meta_batch_size
        loss = loss / args.meta_batch_size
        loss.backward()
        
        if (i + 1) % args.meta_batch_size == 0 or (i + 1) == len(loader):
            if masks is not None:
                apply_mask_grads(model.backbone, masks)
                
            optimizer.step()
            optimizer.zero_grad()
        
        # Multiply loss back for logging
        losses.update(loss.item() * args.meta_batch_size, 1)
        accs.update(acc.item(), 1)
        
        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        if (i + 1) % args.print_freq == 0:
            logger.info(f'Epoch: [{epoch}][{i+1}/{len(loader)}]\t'
                        f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        f'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                        f'Acc {accs.val:.3f} ({accs.avg:.3f})')
            
    return losses.avg, accs.avg

def evaluate(model, loader, args):
    model.eval()
    accs = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if torch.cuda.is_available():
                data = data.cuda()
            
            _, acc = model.proto_loss(data, args.n_way, args.k_shot, args.k_query)
            accs.append(acc.item())
            
    mean, h = mean_confidence_interval(accs)
    return mean, h

def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir = os.path.join(args.output_dir, args.backbone, f"{args.n_way}way_{args.k_shot}shot")
    os.makedirs(args.output_dir, exist_ok=True)
    logger = get_logger(os.path.join(args.output_dir, 'train.log'))
    logger.info(args)
    
    # Dataset & Dataloader
    train_set = MiniImageNet(args.data_root, mode='train', transform=get_transforms('train'))
    train_sampler = CategoriesSampler(train_set.label_to_indices, args.episodes, args.n_way, args.k_shot + args.k_query)
    train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=args.num_workers, pin_memory=True)
    
    val_set = MiniImageNet(args.data_root, mode='val', transform=get_transforms('val'))
    val_sampler = CategoriesSampler(val_set.label_to_indices, args.episodes, args.n_way, args.k_shot + args.k_query)
    val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=args.num_workers, pin_memory=True)
    
    test_set = MiniImageNet(args.data_root, mode='test', transform=get_transforms('test'))
    test_sampler = CategoriesSampler(test_set.label_to_indices, args.test_episodes, args.n_way, args.k_shot + args.k_query) # args.test_episodes for test
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=args.num_workers, pin_memory=True)
    
    # Model
    if args.backbone == 'conv4':
        backbone = conv4()
    else:
        backbone = resnet12()
        
    model = ProtoNet(backbone)
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Save initialization
    torch.save(model.state_dict(), os.path.join(args.output_dir, 'W_init.pth'))
    
    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
    
    # --------------------------------------------------------------------------
    # 1. Dense Training
    # --------------------------------------------------------------------------
    logger.info("Starting Dense Training...")
    best_acc = 0
    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch, args, logger)
        
        if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
            val_acc, val_h = evaluate(model, val_loader, args)
            
            is_best = val_acc > best_acc
            best_acc = max(val_acc, best_acc)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }, is_best, filename='checkpoint_dense.pth', best_filename='model_dense_best.pth', folder=args.output_dir)
            
            logger.info(f'Epoch {epoch}: Train Loss {train_loss:.4f} Acc {train_acc:.2f} | Val Acc {val_acc:.2f} +/- {val_h:.2f}')
        else:
            scheduler.step() # Step scheduler even if val skipped (usually based on epoch)
            logger.info(f'Epoch {epoch}: Train Loss {train_loss:.4f} Acc {train_acc:.2f}')
        
    # Load best dense model
    checkpoint = torch.load(os.path.join(args.output_dir, 'model_dense_best.pth'), weights_only=False)
    model.load_state_dict(checkpoint['state_dict'])
    test_acc, test_h = evaluate(model, test_loader, args)
    logger.info(f'Dense Model Test Acc: {test_acc:.2f} +/- {test_h:.2f}')
    
    # --------------------------------------------------------------------------
    # 2. Pruning & Retraining
    # --------------------------------------------------------------------------
    prune_ratios = {'LS': 10, 'MS': 50, 'HS': 90}
    
    for name, ratio in prune_ratios.items():
        logger.info(f"Starting Pruning: {name} ({ratio}%)")
        
        # Calculate masks from trained dense model
        masks = get_masks(model.backbone, ratio)
        
        # Reset to initialization
        model.load_state_dict(torch.load(os.path.join(args.output_dir, 'W_init.pth'), weights_only=False))
        
        # Apply masks (to zero out weights initially)
        apply_masks(model.backbone, masks)
        
        # Verify sparsity
        sparsity = current_sparsity(model.backbone)
        logger.info(f"Sparsity after reset: {sparsity:.2f}%")
        
        # Retrain
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.1)
        
        best_acc_pruned = 0
        for epoch in range(args.epochs):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, epoch, args, logger, masks=masks)
            
            if (epoch + 1) % 5 == 0 or (epoch + 1) == args.epochs:
                val_acc, val_h = evaluate(model, val_loader, args)
                
                if val_acc > best_acc_pruned:
                    best_acc_pruned = val_acc
                    torch.save(model.state_dict(), os.path.join(args.output_dir, f'model_subnet_{ratio}.pth'))
                    
                logger.info(f'[{name}] Epoch {epoch}: Train Acc {train_acc:.2f} | Val Acc {val_acc:.2f} +/- {val_h:.2f}')
            else:
                logger.info(f'[{name}] Epoch {epoch}: Train Acc {train_acc:.2f}')
                
            scheduler.step()
                
        # Test Pruned Model
        model.load_state_dict(torch.load(os.path.join(args.output_dir, f'model_subnet_{ratio}.pth'), weights_only=False))
        test_acc, test_h = evaluate(model, test_loader, args)
        logger.info(f'{name} Model Test Acc: {test_acc:.2f} +/- {test_h:.2f}')

if __name__ == '__main__':
    main()
