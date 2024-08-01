from modules.model.classifiers.classifer_template import ClassifierTemplate


class SimpleFCClassifier(ClassifierTemplate):
    def __init__(self, model_config, training=True):
        super(SimpleFCClassifier, self).__init__(model_config, training=training)
