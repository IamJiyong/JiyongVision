from modules.model.detectors.detector_template import DetectorTemplate


class SSD(DetectorTemplate):
    def __init__(self, model_config, training=True):
        super(SSD, self).__init__(model_config=model_config, training=training)
