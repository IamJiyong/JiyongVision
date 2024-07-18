from modules.model.classifiers.classifers_template import ClassifierTemplate


class ResNet(ClassifierTemplate):
    def __init__(self, model_config, training=True):
        super(ResNet, self).__init__(model_config, training=training)
    
    def forward(self, batch_dict):
        batch_dict = self.backbone(batch_dict)
        batch_dict = self.head(batch_dict)
        if not self.training:
            pred_dict, val_loss = self.post_processing(batch_dict)
            return pred_dict, val_loss
        else:
            loss = self.get_loss(batch_dict)
            return loss

    def get_loss(self, batch_dict):
        loss = self.head.get_loss(batch_dict)
        return loss