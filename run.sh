
# script for running experiments
#!/bin/bash

# 1. run pretrain - minimagenet - conv4 - 5 way 1 shot
python Pretrain/main.py --data-root ../Datasets/Mini-Imagenet --backbone conv4 --n_way 5 --k_shot 1

# 2. run pretrain - minimagenet - conv4 - 5 way 5 shot
python Pretrain/main.py --data-root ../Datasets/Mini-Imagenet --backbone conv4 --n_way 5 --k_shot 5

# 3. run pretrain - minimagenet - resnet12 - 5 way 1 shot
python Pretrain/main.py --data-root ../Datasets/Mini-Imagenet --backbone resnet12 --n_way 5 --k_shot 1

# 4. run pretrain - minimagenet - resnet12 - 5 way 5 shot
python Pretrain/main.py --data-root ../Datasets/Mini-Imagenet --backbone resnet12 --n_way 5 --k_shot 5

# 5. run protonet - minimagenet - conv4 - 5 way 1 shot
python Protonet/main.py --data-root ../Datasets/Mini-Imagenet --backbone conv4 --n_way 5 --k_shot 1

# 6. run protonet - minimagenet - conv4 - 5 way 5 shot
python Protonet/main.py --data-root ../Datasets/Mini-Imagenet --backbone conv4 --n_way 5 --k_shot 5

# 7. run protonet - minimagenet - resnet12 - 5 way 1 shot
python Protonet/main.py --data-root ../Datasets/Mini-Imagenet --backbone resnet12 --n_way 5 --k_shot 1

# 8. run protonet - minimagenet - resnet12 - 5 way 5 shot
python Protonet/main.py --data-root ../Datasets/Mini-Imagenet --backbone resnet12 --n_way 5 --k_shot 5

# crossdomain
# MAML