from torch import nn


class BackboneTemplate(nn.Module):
    def __init__(self, backbone_config):
        super(BackboneTemplate, self).__init__()
        self.backbone_config = backbone_config
        self.name = backbone_config.NAME
    
    def forward(self, batch_dict):
        raise NotImplementedError
    