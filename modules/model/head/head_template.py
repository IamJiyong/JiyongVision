import torch

from modules.utils import loss_utils


class ClassificationHeadTemplate(torch.nn.Module):
    def __init__(self, head_config):
        super(ClassificationHeadTemplate, self).__init__()
        self.criterion = self.build_loss(head_config.LOSS)

    def forward(self, x):
        raise NotImplementedError

    def build_loss(self, loss_config):
        return getattr(loss_utils, loss_config.NAME)(loss_config=loss_config)
