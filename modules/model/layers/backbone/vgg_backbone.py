import torch.nn as nn
from typing import Type, Union, List

from .backbone_template import BackboneTemplate
from ....utils.model_utils import get_updated_weights


class VGGBackbone(BackboneTemplate):
    def __init__(self, backbone_config) -> None:
        super(VGGBackbone, self).__init__(backbone_config)
        
        self.num_layers = backbone_config.NUM_LAYERS
        architecture, self.name = self.get_vgg_params(self.num_layers)
        batch_norm = backbone_config.get('BATCH_NORM', False)
        
        self.features = self._make_layers(architecture, batch_norm)
        
        if backbone_config.PRETRAINED:
            self.load_pretrained_weights()

        self.save_intermediate = backbone_config.get('SAVE_INTERMEDIATE_FEATURES', False)
    
    def _make_layers(self, architecture: List[int], batch_norm: bool = False) -> nn.Sequential:
        layers = []
        in_channels = 3
        for x in architecture:
            if x == 'M':
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                conv2d = nn.Conv2d(in_channels, x, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(x), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = x


        return nn.Sequential(*layers)
    
    def forward(self, batch_dict: dict) -> dict:
        x = batch_dict['img']
        
        block_num = 1
        for i, layer in enumerate(self.features):
            if self.save_intermediate and isinstance(layer, nn.MaxPool2d):
                batch_dict[f'VGG_intermediate_{block_num}'] = x
                block_num += 1
            x = layer(x)
        
        batch_dict['spatial_features'] = x
        return batch_dict
    
    def load_pretrained_weights(self):
        import torchvision
        
        weight_name = self.name + '_Weights.DEFAULT'
        model = getattr(torchvision.models, self.name.lower())(weights=weight_name)
        new_weights = get_updated_weights(self.state_dict(), model.state_dict())
        
        self.load_state_dict(new_weights, strict=True)
    
    @staticmethod
    def get_vgg_params(num_layers: int):
        vgg_name = 'VGG{}'.format(num_layers)
        if num_layers == 11:
            return [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], vgg_name
        elif num_layers == 13:
            return [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'], vgg_name
        elif num_layers == 16:
            return [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'], vgg_name
        elif num_layers == 19:
            return [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'], vgg_name
        else:
            raise ValueError("Invalid VGG name: {}".format(vgg_name))
