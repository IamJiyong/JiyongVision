import torch
import torch.nn as nn

from modules.utils import losses


class ClassificationHeadTemplate(nn.Module):
    def __init__(self, head_config):
        super(ClassificationHeadTemplate, self).__init__()
        self.num_classes = head_config.NUM_CLASSES
        self.criterion = self.build_loss(head_config.LOSS)

    def forward(self, x):
        raise NotImplementedError

    def build_loss(self, loss_config, pred_key='scores', target_key='target'):
        return losses.build_loss(loss_config, pred_key, target_key, sum_loss=True)

    def get_loss(self, batch_dict):
        return {'loss': self.criterion(batch_dict)}


class DetectionHeadTemplate(nn.Module):
    def __init__(self, head_config):
        super(DetectionHeadTemplate, self).__init__()
        self.num_classes = head_config.NUM_CLASSES
        self.criterion = self.build_loss(head_config.LOSS)

    def forward(self, x):
        raise NotImplementedError

    def build_loss(self, loss_config, pred_key='pred_scores', target_key='target'):
        return losses.build_loss(loss_config, pred_key, target_key)

    def get_loss(self, batch_dict):
        return {'loss': self.criterion(batch_dict)}