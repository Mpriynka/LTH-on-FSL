import argparse
import os
import copy
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from backbones.conv4 import Conv4
from backbones.resnet12 import ResNet12
from utils.dataloader import CIFARFS, MiniImageNet, CategoriesSampler
from fsl_methods.pretrain import train_pretrain, eval_pretrain
from fsl_methods.protonet import train_protonet, eval_protonet
from prune.prune import get_masks, apply_masks, rewind_weights, check_sparsity

def get_backbone(args):
    if args.backbone == 'conv4':
        return Conv4()
    elif args.backbone == 'resnet12':
        return ResNet12()
    else:
        raise ValueError(f"Unknown backbone: {args.backbone}")

def get_dataset(args, mode='train'):
    if args.dataset == 'cifar-fs':
        # CIFAR-FS is 32x32
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        return CIFARFS(args.data_root, mode=mode, transform=transform)
    elif args.dataset == 'mini-imagenet':
        # MiniImageNet is usually resized to 84x84
        transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
        return MiniImageNet(args.data_root, mode=mode, transform=transform)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar-fs', choices=['cifar-fs', 'mini-imagenet'])
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--backbone', type=str, default='conv4', choices=['conv4', 'resnet12'])
    parser.add_argument('--method', type=str, default='protonet', choices=['pretrain', 'protonet'])
    parser.add_argument('--prune_rate', type=float, default=0.0, help='Sparsity level (e.g. 0.5 for 50%)')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save_dir', type=str, default='./experiments')
    
    # ProtoNet specific
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--q_query', type=int, default=15)
    parser.add_argument('--episodes', type=int, default=100)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)
    
    print(f"Running with args: {args}")
    
    # 1. Initialize Model & Save W_init
    model = get_backbone(args).to(device)
    w_init = copy.deepcopy(model.state_dict())
    torch.save(w_init, os.path.join(args.save_dir, 'w_init.pth'))
    
    # 2. Train Dense Model
    print("Training Dense Model...")
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    train_set = get_dataset(args, 'train')
    val_set = get_dataset(args, 'val')
    
    if args.method == 'protonet':
        train_sampler = CategoriesSampler(train_set.labels, args.episodes, args.n_way, args.k_shot + args.q_query)
        train_loader = DataLoader(train_set, batch_sampler=train_sampler, num_workers=4)
        val_sampler = CategoriesSampler(val_set.labels, args.episodes, args.n_way, args.k_shot + args.q_query)
        val_loader = DataLoader(val_set, batch_sampler=val_sampler, num_workers=4)
        
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_protonet(model, train_loader, optimizer, epoch, args.n_way, args.k_shot, args.q_query, device)
            val_acc = eval_protonet(model, val_loader, args.n_way, args.k_shot, args.q_query, device)
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Train Acc {train_acc:.4f}, Val Acc {val_acc:.4f}")
            
    elif args.method == 'pretrain':
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_set, batch_size=64, shuffle=False, num_workers=4)
        
        for epoch in range(1, args.epochs + 1):
            train_loss = train_pretrain(model, train_loader, optimizer, epoch, device)
            val_acc = eval_pretrain(model, val_loader, device)
            print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val Acc {val_acc:.4f}")

    # Save Dense Model
    torch.save(model.state_dict(), os.path.join(args.save_dir, 'w_final_dense.pth'))
    
    # 3. Prune
    if args.prune_rate > 0:
        print(f"Pruning with rate {args.prune_rate}...")
        masks = get_masks(model, args.prune_rate)
        torch.save(masks, os.path.join(args.save_dir, 'masks.pth'))
        
        # 4. Rewind to W_init
        print("Rewinding to W_init...")
        rewind_weights(model, w_init, masks)
        check_sparsity(model)
        
        # 5. Retrain Sparse Model
        print("Retraining Sparse Model...")
        optimizer = optim.Adam(model.parameters(), lr=args.lr) # Reset optimizer
        
        for epoch in range(1, args.epochs + 1):
            # Ensure masks are applied after each step (or use hooks, but simple re-application works for now)
            if args.method == 'protonet':
                train_loss, train_acc = train_protonet(model, train_loader, optimizer, epoch, args.n_way, args.k_shot, args.q_query, device)
                apply_masks(model, masks) # Enforce sparsity
                val_acc = eval_protonet(model, val_loader, args.n_way, args.k_shot, args.q_query, device)
                print(f"Sparse Epoch {epoch}: Train Loss {train_loss:.4f}, Val Acc {val_acc:.4f}")
            elif args.method == 'pretrain':
                train_loss = train_pretrain(model, train_loader, optimizer, epoch, device)
                apply_masks(model, masks)
                val_acc = eval_pretrain(model, val_loader, device)
                print(f"Sparse Epoch {epoch}: Train Loss {train_loss:.4f}, Val Acc {val_acc:.4f}")
                
        torch.save(model.state_dict(), os.path.join(args.save_dir, 'w_final_sparse.pth'))

if __name__ == '__main__':
    main()
