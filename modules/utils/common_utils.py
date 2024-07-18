import torch

def horizontal_flip(batch_dict):
    batch_dict['img'] = torch.flip(batch_dict['img'], [-1])
    return batch_dict