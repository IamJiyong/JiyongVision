import torch.nn as nn

from .head_template import ClassificationHeadTemplate


class SimpleFCHead(ClassificationHeadTemplate):
    def __init__(self, head_config):
        super(SimpleFCHead, self).__init__(head_config)
        self.fc = nn.Linear(head_config.IN_DIM, self.num_classes)

    def forward(self, batch_dict):
        x = batch_dict['embedding']
        x = self.fc(x)
        batch_dict['scores'] = x
        return batch_dict