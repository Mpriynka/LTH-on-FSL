
# script for running experiments
#!/bin/bash

# 1. run pretrain - conv4 - 5 way 1 shot
python3 Pretrain/main.py --backbone conv4 --n_way 5 --k_shot 1

# 2. run pretrain - conv4 - 5 way 5 shot
python3 Pretrain/main.py --backbone conv4 --n_way 5 --k_shot 5

# 3. run pretrain - resnet12 - 5 way 1 shot
python3 Pretrain/main.py --backbone resnet12 --n_way 5 --k_shot 1

# 4. run pretrain - resnet12 - 5 way 5 shot
python3 Pretrain/main.py --backbone resnet12 --n_way 5 --k_shot 5

# 5. run protonet - conv4 - 5 way 1 shot
python3 Protonet/main.py --backbone conv4 --n_way 5 --k_shot 1

# 6. run protonet - conv4 - 5 way 5 shot
python3 Protonet/main.py --backbone conv4 --n_way 5 --k_shot 5

# 7. run protonet - resnet12 - 5 way 1 shot
python3 Protonet/main.py --backbone resnet12 --n_way 5 --k_shot 1

# 8. run protonet - resnet12 - 5 way 5 shot
python3 Protonet/main.py --backbone resnet12 --n_way 5 --k_shot 5

# crossdomain
# MAML