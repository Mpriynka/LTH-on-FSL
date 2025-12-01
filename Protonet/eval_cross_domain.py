import argparse
import os
import torch
from torch.utils.data import DataLoader
from cub import CUB
from mini_imagenet import CategoriesSampler, get_transforms
from backbone.conv4 import conv4
from backbone.resnet12 import resnet12
from protonet import ProtoNet
from utils import set_seed, mean_confidence_interval

def parse_args():
    parser = argparse.ArgumentParser(description='Cross-Domain Evaluation (MiniImageNet -> CUB)')
    parser.add_argument('--data-root', type=str, default='Datasets', help='Path to dataset root')
    parser.add_argument('--backbone', type=str, default='conv4', choices=['conv4', 'resnet12'], help='Backbone model')
    parser.add_argument('--model-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('--n_way', type=int, default=5, help='Number of classes per episode')
    parser.add_argument('--k_shot', type=int, default=1, help='Number of support samples per class')
    parser.add_argument('--k_query', type=int, default=15, help='Number of query samples per class')
    parser.add_argument('--test-episodes', type=int, default=2000, help='Number of episodes for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    return parser.parse_args()

def evaluate(model, loader, args):
    model.eval()
    accs = []
    
    with torch.no_grad():
        for i, (data, _) in enumerate(loader):
            if torch.cuda.is_available():
                data = data.cuda()
            
            _, acc = model.proto_loss(data, args.n_way, args.k_shot, args.k_query)
            accs.append(acc.item())
            
            if (i+1) % 100 == 0:
                print(f"Episode {i+1}/{args.test_episodes}")
            
    mean, h = mean_confidence_interval(accs)
    return mean, h

def main():
    args = parse_args()
    set_seed(args.seed)
    
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)
        print(f"Using GPU {args.gpu}")
    
    # Dataset & Dataloader
    print(f"Loading CUB dataset from {args.data_root}...")
    test_set = CUB(args.data_root, mode='test', transform=get_transforms('test'))
    test_sampler = CategoriesSampler(test_set.label_to_indices, args.test_episodes, args.n_way, args.k_shot + args.k_query)
    test_loader = DataLoader(test_set, batch_sampler=test_sampler, num_workers=8, pin_memory=True)
    
    # Model
    print(f"Loading {args.backbone} model from {args.model_path}...")
    if args.backbone == 'conv4':
        backbone = conv4()
    else:
        backbone = resnet12()
        
    model = ProtoNet(backbone)
    if torch.cuda.is_available():
        model = model.cuda()
        
    # Load Checkpoint
    checkpoint = torch.load(args.model_path, map_location='cpu', weights_only=False)
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint # Handle cases where only state_dict was saved
        
    # Handle key mismatch (Pretrain vs Protonet)
    # Protonet wraps backbone, so keys are 'backbone.layer1...'
    # Pretrain is just backbone, so keys are 'layer1...'
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('backbone.'):
            new_state_dict[k] = v
        else:
            new_state_dict[f'backbone.{k}'] = v
            
    # Load with strict=False to ignore potential shape mismatches in fc layer (if any)
    # or if Protonet has extra params (unlikely)
    msg = model.load_state_dict(new_state_dict, strict=False)
    print(f"Loaded model with msg: {msg}")
    
    # Evaluate
    print("Starting evaluation...")
    mean, h = evaluate(model, test_loader, args)
    
    print(f"Result: {mean:.2f} +/- {h:.2f}")
    
    # Save results
    result_file = os.path.join(os.path.dirname(args.model_path), 'cub_eval_results.txt')
    with open(result_file, 'a') as f:
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Settings: {args.n_way}-way {args.k_shot}-shot\n")
        f.write(f"Accuracy: {mean:.2f} +/- {h:.2f}\n")
        f.write("-" * 20 + "\n")
    print(f"Results saved to {result_file}")

if __name__ == '__main__':
    main()
