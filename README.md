# CIFAR-10 Model Training

This repository contains scripts for training and testing models on the CIFAR-10 and VOC0712 datasets.

## 1. Data Preparation
To prepare the CIFAR-10 dataset, run the following command:

```bash
python modules/dataset/cifar10.py --data_path ./data/cifar10
```

To prepare the VOC dataset, run the following command:

```bash
python modules/dataset/voc.py --data_path ./data/VOC0712
```

## 2. Training
To train and test the model, run the following command:

```bash
# train a ResNet-18 model on CIFAR-10
python train.py --cfg_file cfgs/cifar10_models/resnet18.yaml
```

```bash
# train a SSD300 model with ResNet-101 backbone on VOC0712
python train.py --cfg_file cfgs/voc0712_models/ssd300_res101.yaml
```

For detailed argument settings, refer to train.py.

After training, the model will autmatically be tested on the test set.

## 3. Testing
To test a specific model, run the following command:

```bash
# test the ResNet-18 model on CIFAR-10
python eval.py --cfg_file cfgs/cifar10_models/resnet18.yaml --ckpt_path ./outputs/cifar10_models/resnet18/${ckpt_name}.pth
```
    
```bash
# test the SSD300 model with ResNet-101 backbone on VOC0712
python eval.py --cfg_file cfgs/voc0712_models/ssd300_res101.yaml --ckpt_path ./outputs/voc0712_models/ssd300_res101/${ckpt_name}.pth
```


The output log, TensorBoard logs, and model checkpoints (pth files) are saved in the outputs directory.
