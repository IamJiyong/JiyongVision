import torch.nn as nn
import torch.nn.functional as F

from .backbone_template import BackboneTemplate


class SimpleFCBackbone(BackboneTemplate):
    def __init__(self, backbone_config):
        super(SimpleFCBackbone, self).__init__(backbone_config)

        in_dim = backbone_config.IN_DIM
        hidden_dims= backbone_config.HIDDEN_DIMS

        self.fc_layers = nn.ModuleList()
        for hidden_dim in hidden_dims:
            self.fc_layers.append(nn.Linear(in_dim, hidden_dim))
            in_dim = hidden_dim
        
        self.activation = getattr(F, backbone_config.ACTIVATION)

    def forward(self, batch_dict):
        x = batch_dict['img']
        x = x.view(x.size(0), -1)
        for layer in self.fc_layers:
            x = self.activation(layer(x))
        batch_dict['embedding'] = x
        return batch_dict