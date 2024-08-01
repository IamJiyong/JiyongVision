import torch
import torch.nn as nn


def load_data_to_gpu(data):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(torch.device('cuda'))
        if isinstance(data[key], list):
            try:
                data[key] = [x.to(torch.device('cuda')) for x in data[key]]
            except:
                pass
    return data


def horizontal_flip(batch_dict):
    batch_dict['img'] = torch.flip(batch_dict['img'], [-1])
    return batch_dict


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            try:
                img, boxes, labels = t(img, boxes, labels)
            except:
                img = t(img)
        if boxes is not None:
            return img, boxes, labels
        else:
            return img

