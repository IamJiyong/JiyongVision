from .vgg_backbone import VGGBackbone
from .resnet_backbone import ResNetBackbone
from .simple_fc_backbone import SimpleFCBackbone
from .conv_backbone import ConvNetBackbone


__all__ = {
    'VGGBackbone': VGGBackbone,
    'ResNetBackbone': ResNetBackbone,
    'SimpleFCBackbone': SimpleFCBackbone,
    'ConvNetBackbone': ConvNetBackbone,
}
