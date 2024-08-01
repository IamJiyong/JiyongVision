import torch.nn as nn

from modules.dataset.augmentation_utils.cls_augmentation import MixUp


class BaseLoss:
    def __init__(self, loss_config, pred_key=None, target_key=None, sum_loss=True):
        self.loss_config = loss_config
        self.pred_key = pred_key
        self.target_key = target_key
        self.sum_loss = sum_loss

        self.criterion = self.build_criterion(loss_config)

    def __call__(self, batch_dict, target_dict=None):
        if batch_dict.get('mixup', False):
            y_a = batch_dict[self.target_key + '_a'].long() if target_dict is None \
                else target_dict[self.target_key + '_a'].long()
            y_b = batch_dict[self.target_key + '_b'].long() if target_dict is None \
                else target_dict[self.target_key + '_b'].long()

            loss = MixUp.get_mixup_loss(self.criterion,
                                        batch_dict[self.pred_key],
                                        y_a, y_b,
                                        batch_dict['lmbd'])
        else:
            scores = batch_dict[self.pred_key]
            y = batch_dict[self.target_key] if target_dict is None else target_dict[self.target_key]

            loss = self.criterion(scores, y)

        if self.sum_loss:
            loss = loss.sum(dim=-1) / batch_dict['batch_size']

        return loss
    
    def build_criterion(self, loss_config):
        params = loss_config.get('PARAMS', {})
        params.update({'reduction': 'none'})
        return getattr(nn, loss_config.NAME)(**params)


class CrossEntropyLoss(BaseLoss):
    def __init__(self, loss_config, pred_key=None, target_key=None, sum_loss=True):
        super(CrossEntropyLoss, self).__init__(loss_config, pred_key, target_key, sum_loss)


class SmoothL1Loss(BaseLoss):
    def __init__(self, loss_config, pred_key=None, target_key=None, sum_loss=True):
        super(SmoothL1Loss, self).__init__(loss_config, pred_key, target_key, sum_loss)

