import torch

from modules.optimization.optimizer import scale_hyperparams_by_batch_size


def build_optimizer(model, optim_config):
    if optim_config.get('ADAPTIVE_PARAMS', False):
        optim_config = scale_hyperparams_by_batch_size(optim_config)
    params = optim_config.get('PARAMS', {})
    return getattr(torch.optim, optim_config.NAME)(model.parameters(), **params)


def build_scheduler(optimizer, scheduler_config):
    params = scheduler_config.get('PARAMS', {})
    return getattr(torch.optim.lr_scheduler, scheduler_config.NAME)(optimizer, **params)