import torch

from .layers import backbone, head
from ..utils.model_utils import get_updated_weights


class BaseNetwork(torch.nn.Module):
    def __init__(self, model_config, training=True):
        super(BaseNetwork, self).__init__()
        self.training = training
        self.model_config = model_config
        self.build_network(model_config.MODULES)

        self.ensemble = False
        self.ensemble_cfg = self.model_config.get('ENSEMBLE', None)
        if self.ensemble_cfg is not None and self.ensemble_cfg.ENABLE:
            self.ensemble = True
            self.ensemble_cfg = self.model_config.ENSEMBLE

    def _forward_impl(self, batch_dict):
        for module_name, module in self._modules.items():
            batch_dict = module(batch_dict)
            
        if not self.training:
            pred_dict, val_loss = self.post_processing(batch_dict)
            return pred_dict, val_loss
        else:
            loss = self.get_loss(batch_dict)
            return loss
        
    def _forward_ensemble_impl(self, batch_dict):
        raise NotImplementedError
    
    def forward(self, batch_dict):
        if self.ensemble and not self.training:
            return self._forward_ensemble_impl(batch_dict)
        else:
            return self._forward_impl(batch_dict)

    def get_loss(self, batch_dict):
        loss_dict = self.head.get_loss(batch_dict)
        return loss_dict

    def build_network(self, module_configs):
        for module_name, module_cfg in module_configs.items():
            module_name = module_name.lower()
            if not isinstance(module_cfg, list):
                module_cfg = [module_cfg]

            module_list = []
            for i, config in enumerate(module_cfg):
                module = getattr(self, 'build_' + module_name)(config)
                module_list.append(module)
            if len(module_list) > 1:
                module = torch.nn.Sequential(*module_list)
            else:
                module = module_list[0]
            self.add_module(module_name, module)

    def build_backbone(self, backbone_config):
        return backbone.__all__[backbone_config.NAME](backbone_config)

    def build_head(self, head_config):
        return head.__all__[head_config.NAME](head_config)

    def post_processing(self, batch_dict):
        raise NotImplementedError
    
    def load_ckpt_with_optimizer(self, path, optimizer, lr_scheduler, strict=True):
        state_disk = torch.load(path, map_location='cpu')
        self._load_state_dict(state_disk['model_state_dict'], strict=strict)
        optimizer.load_state_dict(state_disk['optimizer_state_dict'])
        lr_scheduler.load_state_dict(state_disk['lr_scheduler_state_dict'])

        return state_disk['epoch'], state_disk['iter']

    def load_ckpt(self, path, strict=True):
        state_disk = torch.load(path, map_location='cpu')
        self._load_state_dict(state_disk['model_state_dict'], strict=strict)
    
    def _load_state_dict(self, model_state_disk, *, strict=True):
        state_dict = self.state_dict()
        update_model_state = get_updated_weights(state_dict, model_state_disk)

        if strict:
            self.load_state_dict(update_model_state)
        else:
            state_dict.update(update_model_state)
            self.load_state_dict(state_dict)
    
    def save_ckpt(self, path, optimizer=None, lr_scheduler=None, epoch=None, it=None):
        state_dict = {}
        state_dict['model_state_dict'] = self.state_dict()
        if optimizer is not None:
            state_dict['optimizer_state_dict'] = optimizer.state_dict()
        if lr_scheduler is not None:
            state_dict['lr_scheduler_state_dict'] = lr_scheduler.state_dict()
        if epoch is not None:
            state_dict['epoch'] = epoch
        if it is not None:
            state_dict['iter'] = it
        torch.save(state_dict, path)
