from modules.model.classifiers.vgg import VGGNet
from modules.model.classifiers.resnet import ResNet
from modules.model.classifiers.simple_fc_classifier import SimpleFCClassifier

from modules.model.detectors.ssd import SSD

__all__ = {
    'VGGNet': VGGNet,
    'ResNet': ResNet,
    'SimpleFCClassifier': SimpleFCClassifier,
    'SSD': SSD,
}

def build_network(model_config):
    model = __all__[model_config.NAME](model_config)
    return model
