import torch

from ....utils.box_utils import batch_iou


class TargetAssigner:
    def __init__(self, box_coder, proposal_matcher):
        self.box_coder = box_coder
        self.proposal_matcher = proposal_matcher


    def __call__(self, batch_dict, anchors):
        gt_boxes = batch_dict['gt_boxes']
        bbox_regression = batch_dict['reg_pred']
        cls_logits = batch_dict['cls_pred']

        match_quality_matrix = batch_iou(gt_boxes, anchors)
        matched_idxs = self.proposal_matcher(match_quality_matrix)

        num_foreground = []

        reg_targets = torch.zeros_like(bbox_regression)
        cls_targets = torch.zeros(cls_logits.shape[:2], dtype=torch.int64, device=cls_logits.device)

        reg_weights = torch.zeros_like(bbox_regression)

        # Match original targets with default boxes
        for i in range(len(gt_boxes)):
            target_boxes = gt_boxes[i][:,:4]
            target_labels = gt_boxes[i][:,4]
            bbox_regression_per_image = bbox_regression[i]
            cls_logits_per_image = cls_logits[i]
            anchors_per_image = anchors[i]
            matched_idxs_per_image = matched_idxs[i]
            
            # produce the matching between boxes and targets
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
            num_foreground.append(foreground_idxs_per_image.numel())

            # Assign regression targets
            matched_gt_boxes_per_image = target_boxes[foreground_matched_idxs_per_image]
            bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
            anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
            target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)

            # Estimate ground truth for class targets
            gt_classes_target = torch.zeros(
                (cls_logits_per_image.size(0),),
                dtype=target_labels.dtype,
                device=target_labels.device,
            )
            gt_classes_target[foreground_idxs_per_image] = target_labels[
                foreground_matched_idxs_per_image
            ] + 1.0
            
            reg_targets[i, foreground_idxs_per_image, :] = target_regression
            reg_weights[i, foreground_idxs_per_image, :] = 1.0
            cls_targets[i] = gt_classes_target
        
        num_foreground = torch.tensor(num_foreground, dtype=torch.int32, device=cls_logits.device)
        target_dict = {
            "reg_targets": reg_targets,
            "cls_targets": cls_targets,
            "num_foreground": num_foreground,
            "reg_weights": reg_weights
        }
        return target_dict