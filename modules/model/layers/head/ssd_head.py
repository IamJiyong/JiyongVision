import torch.nn as nn
import torch
import torchvision.models.detection._utils as det_utils

from torch import Tensor
from typing import List, Dict

from .box_matcher import SSDMatcher
from .head_template import DetectionHeadTemplate
from .anchor_generator import AnchorGenerator
from .target_assigner import TargetAssigner
from modules.utils import losses
from modules.utils.model_utils import L2Norm


class SSDScoringHead(nn.Module):
    def __init__(self, module_list: nn.ModuleList, num_columns: int):
        super().__init__()
        self.module_list = module_list
        self.num_columns = num_columns

    def _get_result_from_module_list(self, x: Tensor, idx: int) -> Tensor:
        """
        This is equivalent to self.module_list[idx](x),
        but torchscript doesn't support this yet
        """
        num_blocks = len(self.module_list)
        if idx < 0:
            idx += num_blocks
        out = x
        for i, module in enumerate(self.module_list):
            if i == idx:
                out = module(x)
        return out

    def forward(self, x: List[Tensor]) -> Tensor:
        all_results = []

        for i, features in enumerate(x):
            results = self._get_result_from_module_list(features, i)

            # Permute output from (N, A * K, H, W) to (N, HWA, K).
            N, _, H, W = results.shape
            results = results.view(N, -1, self.num_columns, H, W)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(N, -1, self.num_columns)  # Size=(N, HWA, K)

            all_results.append(results)

        return torch.cat(all_results, dim=1)


class SSDClassificationHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], num_classes: int):
        cls_logits = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            cls_logits.append(nn.Conv2d(channels, num_classes * anchors, kernel_size=3, padding=1))
        super().__init__(cls_logits, num_classes)


class SSDRegressionHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int]):
        bbox_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            bbox_reg.append(nn.Conv2d(channels, 4 * anchors, kernel_size=3, padding=1))
        super().__init__(bbox_reg, 4)


class SSDHead(DetectionHeadTemplate):
    def __init__(self, head_config):
        super(SSDHead, self).__init__(head_config=head_config)

        self.intermediate_feature_keys = head_config.INTERMEDIATE_FEATURE_KEYS
        num_classes = head_config.NUM_CLASSES
        in_channel = head_config.IN_CHANNEL

        self.l2_norm = L2Norm(in_channel[0], scale=20)
        
        anchor_generator_config = head_config.ANCHOR_GENERATOR
        self.anchor_generator = AnchorGenerator(anchor_generator_config)
        self.num_anchors = self.anchor_generator.num_anchors_per_location()

        self.classification_head = SSDClassificationHead(in_channel, self.num_anchors, num_classes)
        self.regression_head = SSDRegressionHead(in_channel, self.num_anchors)

        proposal_matcher = SSDMatcher(head_config.MATCHING_IOU_THRESH)
        self.box_coder = det_utils.BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))
        self.target_assigner = TargetAssigner(self.box_coder, proposal_matcher)


    def forward(self, batch_dict):
        features = [batch_dict[key] for key in self.intermediate_feature_keys]
        features[0] = self.l2_norm(features[0])

        batch_dict['cls_pred'] = self.classification_head(features)
        batch_dict['reg_pred'] = self.regression_head(features)

        grid_sizes = [torch.tensor(feature.shape[-2:]).to('cuda') for feature in features]
        anchors = self.anchor_generator(batch_dict['img'], grid_sizes)
        batch_dict['anchors'] = anchors

        self.target_dict = self.target_assigner(batch_dict, anchors) if self.training else None
            
        return batch_dict
    

    def build_loss(self, loss_config, pred_key=None, target_key=None):
        positive_fraction = loss_config.POSITIVE_FRACTION
        self.neg_to_pos_ratio = (1.0 - positive_fraction) / positive_fraction

        cls_criterion = losses.build_loss(loss_config.CLS_LOSS,
                                          pred_key='cls_pred',
                                          target_key='cls_targets',
                                          sum_loss=False)
        reg_criterion = losses.build_loss(loss_config.REG_LOSS,
                                          pred_key='reg_pred',
                                          target_key='reg_targets',
                                          sum_loss=False)

        self.cls_weight = loss_config.WEIGHTS.CLS_LOSS
        self.reg_weight = loss_config.WEIGHTS.REG_LOSS
        return {'cls_criterion': cls_criterion, 'reg_criterion': reg_criterion}
    
    
    def get_loss(self, batch_dict: Dict):
        batch_dict['cls_pred'] = batch_dict['cls_pred'].permute(0, 2, 1).contiguous()
        cls_losses = self.criterion['cls_criterion'](batch_dict, self.target_dict)
        
        num_foreground = self.target_dict['num_foreground']
        num_hard_negatives = (self.neg_to_pos_ratio * num_foreground).int()

        negative_loss_mask = self.target_dict['cls_targets'] == 0
        negative_losses = cls_losses * negative_loss_mask.float()
        sorted_negative_losses, _ = torch.sort(negative_losses, descending=True)

        hard_negatives_loss = sum([sorted_negative_losses[i,:num].sum() for i, num in enumerate(num_hard_negatives)])
        positive_losses = cls_losses * (1 - negative_loss_mask.float())
        cls_loss = (positive_losses.sum() + hard_negatives_loss) / num_foreground.sum()

        reg_loss = (self.criterion['reg_criterion'](batch_dict, self.target_dict) * \
            self.target_dict['reg_weights']).sum() / num_foreground.sum()
        
        loss = cls_loss * self.cls_weight + reg_loss * self.reg_weight
        loss_dict = {'cls_loss': cls_loss, 'reg_loss': reg_loss, 'loss': loss}
        return loss_dict
