from modules.model.classifiers.classifer_template import ClassifierTemplate


class VGGNet(ClassifierTemplate):
    def __init__(self, model_config, training=True):
        super(VGGNet, self).__init__(model_config, training=training)
