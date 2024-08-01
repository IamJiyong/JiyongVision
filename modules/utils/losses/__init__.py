from . import loss_utils


def build_loss(loss_config, pred_key='scores', target_key='target', sum_loss=False):
    return getattr(loss_utils, loss_config.NAME)(loss_config, pred_key, target_key, sum_loss)
