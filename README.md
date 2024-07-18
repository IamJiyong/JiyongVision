# CIFAR-10 Model Training

This repository contains scripts for training and testing models on the CIFAR-10 dataset.

## 1. Data Preparation
To prepare the CIFAR-10 dataset, run the following command:

```bash
python modules.dataset.cifar10.py --data_path ./data/cifar10
```

## 2. Training and Testing
To train and test the model, run the following command:

```bash
python train.py --cfg_file cfgs/cifar10_models/resnet18.yaml
```

For detailed argument settings, refer to train.py.

The output log, TensorBoard logs, and model checkpoints (pth files) are saved in the outputs directory.
