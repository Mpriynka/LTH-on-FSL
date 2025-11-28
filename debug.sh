#!/bin/bash

# Debug script for running experiments with minimal parameters

# Create debug directories
mkdir -p ./checkpoints/debug/Pretrain
mkdir -p ./checkpoints/debug/Protonet

echo "Starting Debug Run..."

# 1. run pretrain - conv4 - 5 way 1 shot
# echo "Debug: Pretrain conv4 5-way 1-shot"
# python3 Pretrain/main.py --backbone conv4 --n_way 5 --k_shot 1 --epochs 1 --test_episodes 10 --save_dir ./checkpoints/debug/Pretrain --prune_ratios 50 --print_freq 100

# # 2. run pretrain - conv4 - 5 way 5 shot
# echo "Debug: Pretrain conv4 5-way 5-shot"
# python3 Pretrain/main.py --backbone conv4 --n_way 5 --k_shot 5 --epochs 1 --test_episodes 10 --save_dir ./checkpoints/debug/Pretrain --prune_ratios 50 --print_freq 100

# # 3. run pretrain - resnet12 - 5 way 1 shot
# echo "Debug: Pretrain resnet12 5-way 1-shot"
# python3 Pretrain/main.py --backbone resnet12 --n_way 5 --k_shot 1 --epochs 1 --test_episodes 10 --save_dir ./checkpoints/debug/Pretrain --prune_ratios 50 --print_freq 100

# # 4. run pretrain - resnet12 - 5 way 5 shot
# echo "Debug: Pretrain resnet12 5-way 5-shot"
# python3 Pretrain/main.py --backbone resnet12 --n_way 5 --k_shot 5 --epochs 1 --test_episodes 10 --save_dir ./checkpoints/debug/Pretrain --prune_ratios 50 --print_freq 100

# # 5. run protonet - conv4 - 5 way 1 shot
# echo "Debug: Protonet conv4 5-way 1-shot"
# python3 Protonet/main.py --backbone conv4 --n_way 5 --k_shot 1 --epochs 1 --episodes 10 --output-dir ./checkpoints/debug/Protonet --print-freq 100

# # 6. run protonet - conv4 - 5 way 5 shot
# echo "Debug: Protonet conv4 5-way 5-shot"
# python3 Protonet/main.py --backbone conv4 --n_way 5 --k_shot 5 --epochs 1 --episodes 10 --output-dir ./checkpoints/debug/Protonet --print-freq 100

# # 7. run protonet - resnet12 - 5 way 1 shot
# echo "Debug: Protonet resnet12 5-way 1-shot"
# python3 Protonet/main.py --backbone resnet12 --n_way 5 --k_shot 1 --epochs 1 --episodes 10 --output-dir ./checkpoints/debug/Protonet --print-freq 100

# 8. run protonet - resnet12 - 5 way 5 shot
echo "Debug: Protonet resnet12 5-way 5-shot"
python3 Protonet/main.py --backbone resnet12 --n_way 5 --k_shot 5 --epochs 1 --episodes 10 --output-dir ./checkpoints/debug/Protonet --print-freq 100

echo "Debug Run Completed."
