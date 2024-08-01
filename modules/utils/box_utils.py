# -*- coding: utf-8 -*-
import torch


def batch_iou(boxes_1, boxes_2):
    """
    Calculate IoU for two sets of bounding boxes.
    
    Args:
    - boxes_1 (torch.Tensor): Bounding boxes of shape (B, N, 4)
    - boxes_2 (torch.Tensor): Bounding boxes of shape (B, M, 4)
    
    Returns:
    - iou_map (torch.Tensor): IoU map of shape (B, N, M)
    """
    B, N, _ = boxes_1.shape
    _, M, _ = boxes_2.shape
    
    # Expand dimensions for broadcasting
    boxes_1_exp = boxes_1.unsqueeze(2)  # (B, N, 1, 4)
    boxes_2_exp = boxes_2.unsqueeze(1)  # (B, 1, M, 4)
    
    # Calculate intersection
    x1_int = torch.max(boxes_1_exp[..., 0], boxes_2_exp[..., 0])
    y1_int = torch.max(boxes_1_exp[..., 1], boxes_2_exp[..., 1])
    x2_int = torch.min(boxes_1_exp[..., 2], boxes_2_exp[..., 2])
    y2_int = torch.min(boxes_1_exp[..., 3], boxes_2_exp[..., 3])
    
    inter_w = torch.clamp(x2_int - x1_int, min=0)
    inter_h = torch.clamp(y2_int - y1_int, min=0)
    inter_area = inter_w * inter_h
    
    # Calculate areas of both boxes
    area_1 = (boxes_1_exp[..., 2] - boxes_1_exp[..., 0]) * (boxes_1_exp[..., 3] - boxes_1_exp[..., 1])
    area_2 = (boxes_2_exp[..., 2] - boxes_2_exp[..., 0]) * (boxes_2_exp[..., 3] - boxes_2_exp[..., 1])
    
    # Calculate union area
    union_area = area_1 + area_2 - inter_area
    
    # Calculate IoU
    iou_map = inter_area / torch.clamp(union_area, min=1e-6)  # Avoid division by zero

    return iou_map