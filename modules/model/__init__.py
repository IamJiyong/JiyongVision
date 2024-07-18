from modules.model.classifiers.resnet import ResNet

__all__ = {
    'ResNet': ResNet,
}

def build_network(model_config):
    model = __all__[model_config.NAME](model_config)
    return model