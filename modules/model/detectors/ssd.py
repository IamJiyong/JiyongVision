from modules.model.detectors.detector_template import DetectorTemplate
from modules.model.layers.backbone.vgg_backbone import VGGBackbone

class SSD(DetectorTemplate):
    def __init__(self, model_config, training=True):
        super(SSD, self).__init__(model_config=model_config, training=training)

        if isinstance(self.backbone[0], VGGBackbone) and self.backbone[0].num_layers == 16:
            self.backbone[0].features[16].ceil_mode = True
