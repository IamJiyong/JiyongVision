import numpy as np
import torch
import random

from torchvision.transforms import *


class MixUp(object):
    def __init__(self, alpha=1.0, apply_ratio=1.0):
        self.alpha = alpha
        self.apply_ratio = apply_ratio

    def apply(self, batch_dict):
        if random.random() > self.apply_ratio:
            batch_dict['mixup'] = False
            return batch_dict
        
        img = batch_dict['img']
        target = batch_dict['target']

        lmbd = np.random.beta(self.alpha, self.alpha)
        rand_index = torch.randperm(img.size()[0])
        mixed_img = lmbd * img + (1 - lmbd) * img[rand_index, :]

        batch_dict['img'] = mixed_img
        batch_dict['target_a'] = batch_dict.pop('target')
        batch_dict['target_b'] = target[rand_index]
        batch_dict['lmbd'] = lmbd
        batch_dict['mixup'] = True

        return batch_dict
    
    @staticmethod
    def get_mixup_loss(criterion, pred, target_a, target_b, lmbd):
        return lmbd * criterion(pred, target_a) + (1 - lmbd) * criterion(pred, target_b)
