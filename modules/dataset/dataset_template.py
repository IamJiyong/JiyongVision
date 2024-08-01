import torch
from torch.utils.data import Dataset

from functools import partial
from pathlib import Path
from collections import defaultdict

from modules.dataset.augmentation_utils.augmentor import Augmentor


class DatasetTemplate(Dataset):
    def __init__(self, root_path, data_config, mode, download=False):
        self.num_classes = data_config.NUM_CLASSES
        self.class_names = data_config.CLASS_NAMES
        self.class_to_ind = dict(zip(self.class_names, range(len(self.class_names))))
        
        self.root = root_path
        self.data_path = root_path / Path(data_config.DATA_PATH)
        self.transform = Augmentor(data_config.TRANSFORMS, mode, task=data_config.TASK)
        self.mixup_transfomer = None

        self.download = download
        if self.download:
            self.download_data()

        self.mode = mode
        self.data, self.targets = self.load_data()

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.data)

    def download_data(self):
        raise NotImplementedError

    def load_data(self):
        raise NotImplementedError
    
    def get_collate_fn(self):
        return partial(DatasetTemplate.collate_batch, mixup=self.transform.mixup_transfomer)
    
    @staticmethod
    def collate_batch(batch_list=None, mixup=None):
        if batch_list is None:
            return partial(DatasetTemplate.collate_batch, mixup=mixup)
        
        data_dict = defaultdict(list)
        ret = {}
        for cur_sample in batch_list:
            for key, value in cur_sample.items():
                data_dict[key].append(value)
        batch_size = len(batch_list)

        for key,val in data_dict.items():
            if key == 'img':
                ret[key] = torch.stack(val, dim=0)
            elif key == 'target':
                ret[key] = torch.tensor(val)
            elif key == 'gt_boxes':
                max_gt_num = max([len(x) for x in val])
                gt_boxes = torch.zeros(batch_size, max_gt_num, val[0].shape[-1])
                for i, boxes in enumerate(val):
                    gt_boxes[i, :len(boxes)] = torch.tensor(boxes)
                ret[key] = gt_boxes
                # for i in range(batch_size):
                #     val[i] = torch.tensor(val[i])
                # ret[key] = val
            elif key in ['img_id', 'original_size']:
                ret[key] = val
            else:
                raise NotImplementedError
        
        if mixup is not None:
            ret = mixup.apply(ret)
        
        ret['batch_size'] = batch_size
        return ret
