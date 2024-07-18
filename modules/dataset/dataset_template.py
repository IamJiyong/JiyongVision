import torch
import torchvision.transforms as T
import torchvision.transforms.v2 as T2

from pathlib import Path
from collections import defaultdict


class DatasetTemplate(torch.utils.data.Dataset):
    def __init__(self, root_path, data_config, mode, download=False):
        self.root = root_path
        self.data_path = root_path / Path(data_config["DATA_PATH"])
        self.transform = self.get_transform(data_config["TRANSFORMS"], mode)
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
    
    @staticmethod
    def collate_batch(batch_list):
        data_dict = defaultdict(list)
        ret = {}
        for cur_sample in batch_list:
            for key, value in cur_sample.items():
                data_dict[key].append(value)
        batch_size = len(batch_list)

        for key,val in data_dict.items():
            if key is 'img':
                ret[key] = torch.stack(val, dim=0)
            elif key is 'target':
                ret[key] = torch.tensor(val)
            else:
                raise NotImplementedError
        
        ret['batch_size'] = batch_size
        return ret

    def get_transform(self, transform_cfg, mode):
        transform = []
        for cfg in transform_cfg[mode.upper()]:
            name = cfg["NAME"]
            if name == 'MixUp':
                self.mixup_transfomer = T2.MixUp(**cfg["PARAMS"])
                continue
            params = cfg.get("PARAMS", {})
            transform.append(getattr(T, name)(**params))
        return T.Compose(transform)
    
    def mixup(self, batch_dict):
        if self.mixup_transfomer is None:
            return batch_dict
        batch_dict['img'], batch_dict['target'] = self.mixup_transfomer(batch_dict['img'], batch_dict['target'])
        return batch_dict