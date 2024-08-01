import torch

from torchvision.models.detection.anchor_utils import DefaultBoxGenerator


class AnchorGenerator(DefaultBoxGenerator):
    def __init__(self, anchor_generator_config):
        aspect_ratios = anchor_generator_config.ASPECT_RATIOS
        scales = anchor_generator_config.SCALES
        steps = anchor_generator_config.STEPS

        clip = anchor_generator_config.get('CLIP', False)
        min_ratio = anchor_generator_config.get('MIN_RATIO', None)
        max_ratio = anchor_generator_config.get('MAX_RATIO', None)
        
        super(AnchorGenerator, self).__init__(aspect_ratios=aspect_ratios,
                                              min_ratio=min_ratio,
                                              max_ratio=max_ratio,
                                              scales=scales,
                                              steps=steps,
                                              clip=clip)
    
    def forward(self, images, grid_sizes):
        B, _, H, W = images.shape
        image_size = torch.tensor([H, W], dtype=torch.float32).to('cuda')
        
        default_boxes = self._grid_default_boxes(grid_sizes, image_size)

        # (cx, cy, w, h) -> (xmin, ymin, xmax, ymax)
        default_boxes[:, :2] -= default_boxes[:, 2:] / 2
        default_boxes[:, 2:] += default_boxes[:, :2]
        if self.clip:
            default_boxes.clamp_(min=0, max=1)

        default_boxes = default_boxes.unsqueeze(0).repeat(B, 1, 1)
        return default_boxes

    
    # Default Boxes calculation based on page 6 of SSD paper
    def _grid_default_boxes(self, grid_sizes, image_size, dtype=torch.float32):
        device = image_size.device

        default_boxes = []
        for k, f_k in enumerate(grid_sizes):
            # Now add the default boxes for each width-height pair
            if self.steps is not None:
                x_f_k = image_size[1] / self.steps[k]
                y_f_k = image_size[0] / self.steps[k]
            else:
                y_f_k, x_f_k = f_k

            width_range = torch.arange(0, f_k[0], dtype=dtype, device=device)
            height_range = torch.arange(0, f_k[1], dtype=dtype, device=device)
            
            shifts_x = ((width_range + 0.5) / x_f_k).to(dtype=dtype)
            shifts_y = ((height_range + 0.5) / y_f_k).to(dtype=dtype)
            shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
            shift_x = shift_x.reshape(-1)
            shift_y = shift_y.reshape(-1)

            shifts = torch.stack((shift_x, shift_y) * len(self._wh_pairs[k]), dim=-1).reshape(-1, 2)
            # Clipping the default boxes while the boxes are encoded in format (cx, cy, w, h)
            _wh_pair = self._wh_pairs[k].clamp(min=0, max=1) if self.clip else self._wh_pairs[k]
            wh_pairs = _wh_pair.repeat((f_k[0] * f_k[1]), 1).to(device)

            default_box = torch.cat((shifts, wh_pairs), dim=1)

            default_boxes.append(default_box)

        return torch.cat(default_boxes, dim=0)