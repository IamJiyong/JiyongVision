import torch

from modules.model.network_template import BaseNetwork
from modules.utils.common_utils import horizontal_flip


class ClassifierTemplate(BaseNetwork):
    def __init__(self, model_config, training=True):
        super(ClassifierTemplate, self).__init__(model_config=model_config, training=training)
        self.task = 'classification'

    def _forward_ensemble_impl(self, batch_dict): # TODO: hard coded
        assert self.ensemble and not self.training

        pred_dict_list = []

        pred_dict, _ = self._forward_impl(batch_dict)
        pred_dict_list.append(pred_dict)

        pred_dict, _ = self._forward_impl(horizontal_flip(batch_dict))
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
    
    def post_processing(self, batch_dict):
        assert not self.training

        pred_dict = {}
        scores = batch_dict['scores']
        pred_dict['pred_scores'] = torch.softmax(scores, dim=1)
        pred_dict['pred_classes'] = torch.argmax(scores, dim=1)

        val_loss_dict = self.get_loss(batch_dict)

        return pred_dict, val_loss_dict
