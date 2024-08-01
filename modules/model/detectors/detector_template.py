import torch
import torch.nn.functional as F

from torchvision.ops import boxes as box_ops

from modules.model.network_template import BaseNetwork


class DetectorTemplate(BaseNetwork):
    def __init__(self, model_config, training=True):
        super(DetectorTemplate, self).__init__(model_config=model_config, training=training)
        self.task = 'detection'
        
        post_processing_config = self.model_config.POST_PROCESSING
        self.topk_candidates = post_processing_config.TOPK_CANDIDATES
        self.score_thresh = post_processing_config.SCORE_THRESH
        self.nms_thresh = post_processing_config.NMS_THRESH


    def post_processing(self, batch_dict):
        pred_dicts = []

        reg_pred = batch_dict['reg_pred']
        cls_pred = batch_dict['cls_pred']

        batch_size = batch_dict['batch_size']
        for i in range(batch_size):
            reg_pred_per_image = reg_pred[i]
            cls_pred_per_image = cls_pred[i]
            
            # Decode boxes
            anchors = batch_dict['anchors'][i]
            decoded_boxes = self.head.box_coder.decode_single(reg_pred_per_image, anchors)
            
            img_shape = batch_dict['original_size'][i] # H, W
            decoded_boxes[:, 0::2] = torch.clamp(decoded_boxes[:, 0::2] * img_shape[1], min=0, max=img_shape[1])
            decoded_boxes[:, 1::2] = torch.clamp(decoded_boxes[:, 1::2] * img_shape[0], min=0, max=img_shape[0])

            # Filter boxes based on scores
            scores = F.softmax(cls_pred_per_image, dim=1)
            
            image_boxes = []
            image_scores = []
            image_labels = []

            for j in range(1, scores.shape[1]):
                score = scores[:, j]

                keep_idx = score > self.score_thresh
                score = score[keep_idx]
                box = decoded_boxes[keep_idx]

                num_topk = min(self.topk_candidates, score.size(0))
                score, idx = score.topk(num_topk)
                box = box[idx]

                image_boxes.append(box)
                image_scores.append(score)
                # use j-1 if background is not included in the class list
                image_labels.append(torch.full_like(score, fill_value=j-1, dtype=torch.int64, device='cuda'))

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)
            
            # NMS
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[:self.topk_candidates]

            pred_dict = {
                'pred_boxes': image_boxes[keep],
                'pred_scores': image_scores[keep],
                'pred_labels': image_labels[keep]
            }
            pred_dicts.append(pred_dict)

        return pred_dicts, None            
