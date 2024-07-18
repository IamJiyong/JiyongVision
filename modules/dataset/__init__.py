import torch
from torch.utils.data import DataLoader

from modules.dataset.cifar10 import Cifar10

__all__ = {
    'Cifar10': Cifar10,
}


def build_dataloader(root_dir, data_config, args, mode):
    dataset = __all__[data_config.DATASET](root_dir,
                                           data_config,
                                           mode=mode)
    dataloader = DataLoader(dataset,
                            batch_size=args.batch_size,
                            shuffle=(mode=='train'),
                            num_workers=args.workers,
                            collate_fn=dataset.collate_batch)
    return dataloader

def load_data_to_gpu(data):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(torch.device('cuda'))
    return data