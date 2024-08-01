from .vgg_head import VGGHead
from .resnet_head import ResNetHead
from .simple_fc_head import SimpleFCHead
from .ssd_head import SSDHead


__all__ = {
    'VGGHead': VGGHead,
    'ResNetHead': ResNetHead,
    'SimpleFCHead': SimpleFCHead,
    'SSDHead': SSDHead,
}