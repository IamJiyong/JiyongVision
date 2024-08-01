import torch
from torch.utils.data import DataLoader

from modules.dataset.cifar10 import Cifar10
from modules.dataset.mnist import MNIST
from modules.dataset.voc0712 import VOC0712

__all__ = {
    'Cifar10': Cifar10,
    'MNIST': MNIST,
    'VOC0712': VOC0712,
}


def build_dataloader(root_dir, data_config, args, mode):
    dataset = __all__[data_config.DATASET](root_dir,
                                           data_config,
                                           mode=mode)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=(mode=='train'),
                            num_workers=args.workers,
                            collate_fn=dataset.get_collate_fn(),)
    return dataloader
