import torch

from functools import partial


def CrossEntropyLoss(loss_config, batch_dict=None):
    if batch_dict is None:
        return partial(CrossEntropyLoss, loss_config)
    scores = batch_dict['scores']
    y = batch_dict['target'].long()
    return torch.nn.CrossEntropyLoss()(scores, y)
