#!/bin/bash

# Base directory for checkpoints
# Base directory for checkpoints
CHECKPOINT_ROOT="checkpoints"
DATA_ROOT="Datasets"

# Iterate over experiment types (Protonet, Pretrain)
for exp_type in "Protonet" "Pretrain"; do
    CHECKPOINT_DIR="$CHECKPOINT_ROOT/$exp_type"
    
    if [ ! -d "$CHECKPOINT_DIR" ]; then
        continue
    fi
    
    echo "Processing Experiment Type: $exp_type"

    # Iterate over backbones
    for backbone in "conv4" "resnet12"; do
        if [ -d "$CHECKPOINT_DIR/$backbone" ]; then
            echo "  Processing backbone: $backbone"
            
            # Iterate over n_way_k_shot directories
            for exp_dir in "$CHECKPOINT_DIR/$backbone"/*; do
                if [ -d "$exp_dir" ]; then
                    echo "    Processing experiment: $exp_dir"
                    
                    # Extract n_way and k_shot from directory name (e.g., 5way_1shot)
                    dirname=$(basename "$exp_dir")
                    n_way=$(echo $dirname | cut -d'_' -f1 | sed 's/way//')
                    k_shot=$(echo $dirname | cut -d'_' -f2 | sed 's/shot//')
                    
                    # Find all .pth files (dense and sparse)
                    for model_path in "$exp_dir"/*.pth; do
                        if [[ "$model_path" == *"W_init.pth"* ]]; then
                            continue
                        fi
                        
                        if [[ "$model_path" == *"optimizer"* ]]; then
                            continue
                        fi

                        echo "      Evaluating model: $model_path"
                        
                        ./venv/bin/python Protonet/eval_cross_domain.py \
                            --data-root "$DATA_ROOT" \
                            --backbone "$backbone" \
                            --model-path "$model_path" \
                            --n_way "$n_way" \
                            --k_shot "$k_shot" \
                            --test-episodes 2000
                    done
                fi
            done
        fi
    done
done
