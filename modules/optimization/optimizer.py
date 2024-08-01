import torch


def scale_hyperparams_by_batch_size(config):
    batch_size = config.BATCH_SIZE
    baseline_bs = config.BASELINE_BATCH_SIZE
    opt_name = config.OPTIMIZER.NAME

    k = batch_size / baseline_bs

    if 'Adam' in opt_name:
        lr = config.OPTIMIZER.get('lr', 0.001)
        beta1, beta2 = config.OPTIMIZER.get('betas', (0.9, 0.999))
        eps = config.OPTIMIZER.get('eps', 1e-8)

        root_k = k ** (1/2)
        new_lr = root_k * lr
        new_beta1 = 1 - k * (1 - beta1)
        new_beta2 = 1 - k * (1 - beta2)
        new_eps = eps / root_k

        config.OPTIMIZER.lr = new_lr
        config.OPTIMIZER.betas = (new_beta1, new_beta2)
        config.OPTIMIZER.eps = new_eps

    elif 'SGD' in opt_name:
        lr = config.OPTIMIZER.get('lr', 0.001)
        config.OPTIMIZER.lr = k * lr

    else:
        raise NotImplementedError

    return config
