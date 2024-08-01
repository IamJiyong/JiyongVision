from modules.dataset.augmentation_utils import det_augmentation as det_aug
from modules.dataset.augmentation_utils import cls_augmentation as cls_aug
from modules.utils.common_utils import Compose


class Augmentor(object):
    def __init__(self, aug_config, mode='train', task='detection'):
        aug_list = []
        configs = aug_config[mode.upper()]
        self.mixup_transfomer = None
        for config in configs:
            aug_name = config.NAME
            params = config.get('PARAMS', {})
            if task == 'detection':
                augmentor = getattr(det_aug, aug_name)(**params)
            elif task == 'classification':
                if aug_name == 'MixUp':
                    self.mixup_transfomer = cls_aug.MixUp(**params)
                    continue
                augmentor = getattr(cls_aug, aug_name)(**params)
            aug_list.append(augmentor)
        self.augment = Compose(aug_list)

    def __call__(self, img, boxes=None, labels=None):
        return self.augment(img, boxes, labels)
