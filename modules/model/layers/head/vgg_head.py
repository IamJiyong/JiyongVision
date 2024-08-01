import torch
import torch.nn as nn
from .head_template import ClassificationHeadTemplate

class VGGHead(ClassificationHeadTemplate):
    def __init__(self, head_config):
        super(VGGHead, self).__init__(head_config)
        self.avg_pool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(head_config.IN_DIM * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, self.num_classes),
        )

    def forward(self, batch_dict):
        spatial_features = batch_dict['spatial_features']
        x = self.avg_pool(spatial_features)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        batch_dict['scores'] = x
        return batch_dict
