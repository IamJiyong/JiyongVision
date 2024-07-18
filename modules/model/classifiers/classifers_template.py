import torch

from modules.model import backbone, head
from modules.utils.common_utils import horizontal_flip


class ClassifierTemplate(torch.nn.Module):
    def __init__(self, model_config, training=True):
        super(ClassifierTemplate, self).__init__()
        self.model_config = model_config

        self.backbone = self.build_backbone(model_config.BACKBONE)
        self.head = self.build_head(model_config.HEAD)

        self.training = training

        self.ensemble = False
        self.ensemble_cfg = None
        if self.model_config.ENSEMBLE.ENABLE:
            self.ensemble = True
            self.ensemble_cfg = self.model_config.ENSEMBLE

    def forward(self, batch_dict):
        raise NotImplementedError

    def forward_ensemble(self, batch_dict): # TODO: hard coded
        assert self.ensemble and not self.training

        pred_dict_list = []

        pred_dict, _ = self.forward(batch_dict)
        pred_dict_list.append(pred_dict)

        pred_dict, _ = self.forward(horizontal_flip(batch_dict))
        pred_dict_list.append(pred_dict)
        
        ensembled_pred_dict = {}

        if self.ensemble_cfg.METHOD == 'mean':
            scores = torch.stack([pred_dict['pred_scores'] for pred_dict in pred_dict_list], dim=0).mean(dim=0)
            ensembled_pred_dict['pred_scores'] = torch.softmax(scores, dim=1)
            ensembled_pred_dict['pred_classes'] = torch.argmax(ensembled_pred_dict['pred_scores'], dim=1)

        elif self.ensemble_cfg.METHOD == 'vote':
            scores = torch.stack([pred_dict['pred_scores'] for pred_dict in pred_dict_list], dim=0).mean(dim=0)
            ensembled_pred_dict['pred_scores'] = torch.softmax(scores, dim=1)
            pred_classes_list = torch.stack([pred_dict['pred_classes'] for pred_dict in pred_dict_list], dim=0)
            pred_classes_list = pred_classes_list.T
            ensembled_pred_dict['pred_classes'], _ = torch.mode(pred_classes_list, dim=1)

        else:
            raise ValueError("Invalid ensemble method")
        
        batch_dict['scores'] = scores
        loss = self.get_loss(batch_dict)
        
        return ensembled_pred_dict, loss
    
    def get_loss(self, batch_dict):
        raise NotImplementedError

    def build_backbone(self, backbone_config):
        return backbone.__all__[backbone_config.NAME](backbone_config)

    def build_head(self, head_config):
        return head.__all__[head_config.NAME](head_config)

    def post_processing(self, batch_dict):
        assert not self.training

        pred_dict = {}
        scores = batch_dict['scores']
        pred_dict['pred_scores'] = torch.softmax(scores, dim=1)
        pred_dict['pred_classes'] = torch.argmax(scores, dim=1)

        val_loss = self.get_loss(batch_dict)

        return pred_dict, val_loss
    
    def load_ckpt(self, ckpt):
        self.load_state_dict(ckpt['model_state_dict'])
        return self
    
    def save_ckpt(self, path):
        try:
            torch.save({'model_state_dict': self.state_dict()}, path)
        except:
            print("Failed to save model")
