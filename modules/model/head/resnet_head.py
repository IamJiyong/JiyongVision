import torch

from modules.model.head.head_template import ClassificationHeadTemplate


class ResNetHead(ClassificationHeadTemplate):
    def __init__(self, head_config):
        super(ResNetHead, self).__init__(head_config)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc = torch.nn.Linear(head_config.IN_DIM, head_config.NUM_CLASSES)

    def forward(self, batch_dict):
        spatial_features = batch_dict['spatial_features']
        x = self.avg_pool(spatial_features)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        batch_dict['scores'] = x
        return batch_dict

    def get_loss(self, batch_dict):
        return self.criterion(batch_dict)