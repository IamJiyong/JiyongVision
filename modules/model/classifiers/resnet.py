from modules.model.classifiers.classifer_template import ClassifierTemplate


class ResNet(ClassifierTemplate):
    def __init__(self, model_config, training=True):
        super(ResNet, self).__init__(model_config, training=training)
