import torch
import torch.nn as nn


def get_updated_weights(state_dict, model_state_disk):
    update_model_state = {}
    for key, val in model_state_disk.items():
        if key in state_dict and state_dict[key].shape == val.shape:
            update_model_state[key] = val
        elif key in state_dict and state_dict[key].shape != val.shape:
            print(f"Shape mismatch for key {key}, expected {state_dict[key].shape}, got {val.shape}")
        else:
            print(f"Key {key} not found in state_dict")

    return update_model_state


class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = torch.sqrt(x.pow(2).sum(dim=1, keepdim=True)) + self.eps
        x = torch.div(x, norm)
        x = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return x
