# Exploring Lottery Ticket Hypothesis in Few-Shot Learning (LTH-on-FSL)

This repository contains the implementation of the **Lottery Ticket Hypothesis (LTH)** in the context of **Few-Shot Learning (FSL)**. The project aims to reproduce and extend the findings from the paper *"Exploring Lottery Ticket Hypothesis in Few-Shot Learning"* by Yu Xie et al.

The goal is to investigate whether "winning tickets" (sparse subnetworks found via iterative magnitude pruning) exist in Few-Shot Learning models and if they can transfer across domains.

## 📂 Project Structure

The repository is organized as follows:

```
LTH-on-FSL/
├── Pretrain/               # Code for Pretraining-based FSL experiments
│   ├── main.py             # Main entry point for Pretrain experiments
│   ├── mini_imagenet.py    # Dataset loader for Mini-ImageNet
│   ├── pruning.py          # Pruning logic (IMP)
│   └── ...
├── Protonet/               # Code for Prototypical Networks experiments
│   ├── main.py             # Main entry point for ProtoNet experiments
│   ├── protonet.py         # ProtoNet model definition
│   ├── eval_cross_domain.py# Script for cross-domain evaluation (e.g., on CUB)
│   └── ...
├── backbone/               # Shared backbone architectures
│   ├── conv4.py            # 4-layer Convolutional Network
│   └── resnet12.py         # ResNet-12 architecture
├── Datasets/               # Directory for datasets (Mini-ImageNet, CUB, etc.)
├── checkpoints/            # Directory where models and logs are saved
├── run_cross_domain.sh     # Shell script to automate cross-domain evaluation
└── requirements.txt        # Python dependencies
```

## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd LTH-on-FSL
    ```

2.  **Set up the environment:**
    Ensure you have Python 3 installed. It is recommended to use a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3.  **Prepare Datasets:**
    -   **Mini-ImageNet:** Place the Mini-ImageNet dataset in `Datasets/Mini-Imagenet`.
    -   **CUB:** (Optional, for cross-domain) Place the CUB dataset in `Datasets/CUB`.

## 🚀 Usage

The project supports two main training paradigms: **Pretraining** and **Prototypical Networks (ProtoNet)**. Both support finding lottery tickets via Global Pruning.

### 1. Pretraining Experiments

The `Pretrain` module trains a standard classifier on the base classes and then evaluates it using a nearest-neighbor classifier on the novel classes.

**Command:**
```bash
python Pretrain/main.py --backbone resnet12 --gpu 0
```

**Key Arguments:**
-   `--backbone`: Architecture to use (`resnet12` or `conv4`).
-   `--prune_ratios`: List of sparsity levels to evaluate (e.g., `10 50 90` for 10%, 50%, 90% pruned).
-   `--save_dir`: Directory to save checkpoints (default: `./checkpoints/Pretrain`).
-   `--n_way`, `--k_shot`: Configuration for episodic evaluation.

**Workflow:**
1.  Train a dense network.
2.  Evaluate the dense network.
3.  Prune the network to specified ratios.
4.  Retrain the sparse subnetworks (Winning Tickets).
5.  Evaluate the sparse subnetworks.

### 2. Prototypical Networks (ProtoNet)

The `Protonet` module trains the model using the episodic Prototypical Loss.

**Command:**
```bash
python Protonet/main.py --backbone conv4 --n_way 5 --k_shot 1 --gpu 0
```

**Key Arguments:**
-   `--backbone`: Architecture to use (`resnet12` or `conv4`).
-   `--n_way`: Number of classes per episode (default: 5).
-   `--k_shot`: Number of support samples per class (default: 1).
-   `--k_query`: Number of query samples per class (default: 15).
-   `--episodes`: Number of episodes per epoch.
-   `--output-dir`: Directory to save checkpoints (default: `./checkpoints/Protonet`).

**Workflow:**
Similar to Pretrain, it trains a dense ProtoNet, then iteratively prunes and retrains to find sparse ProtoNets.

### 3. Cross-Domain Evaluation

To evaluate the transferability of the learned lottery tickets to a different domain (e.g., from Mini-ImageNet to CUB), use the provided shell script.

**Command:**
```bash
bash run_cross_domain.sh
```

This script will:
1.  Iterate through all trained checkpoints in `checkpoints/`.
2.  Run `Protonet/eval_cross_domain.py` for each model (dense and sparse).
3.  Report the accuracy on the CUB dataset.

## 📊 Results & Checkpoints

Checkpoints are automatically saved in the `checkpoints/` directory, organized by experiment type, backbone, and shot configuration.

Example structure:
```
checkpoints/
├── Protonet/
│   ├── conv4/
│   │   └── 5way_1shot/
│   │       ├── model_dense_best.pth
│   │       ├── model_subnet_10.pth
│   │       └── train.log
└── Pretrain/
    └── resnet12/
        └── ...
```

## 📝 Citation

If you use this code, please cite the original paper:

```bibtex
@article{xie2020exploring,
  title={Exploring Lottery Ticket Hypothesis in Few-Shot Learning},
  author={Xie, Yu and others},
  journal={arXiv preprint arXiv:2000.00000},
  year={2020}
}
```
