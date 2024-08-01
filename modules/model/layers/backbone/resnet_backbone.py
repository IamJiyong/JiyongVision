'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Any, Callable, List, Optional, Type, Union

from .backbone_template import BackboneTemplate
from ....utils.model_utils import get_updated_weights

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out



class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition" https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetBackbone(BackboneTemplate):
    def __init__(self, backbone_config) -> None:
        super(ResNetBackbone, self).__init__(backbone_config)
        groups = backbone_config.get('GROUPS', 1)
        width_per_group = backbone_config.get('WIDTH_PER_GROUP', 64)
        replace_stride_with_dilation = backbone_config.get('REPLACE_STRIDE_WITH_DILATION', None)

        self.num_layers = backbone_config.NUM_LAYERS
        params, self.name = self.get_resnet_params(backbone_config.NUM_LAYERS)
        block = params['BLOCK']
        layers = params['NUM_BLOCKS']

        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.init_downsample = backbone_config.get('INIT_DOWNSAMPLE', True)
        if self.init_downsample:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) if self.init_downsample else None
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2])

        if backbone_config.PRETRAINED:
            self.load_pretrained_weights()
        
        self.save_intermediate = backbone_config.get('SAVE_INTERMEDIATE_FEATURES', False)


    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, self.base_width, previous_dilation, norm_layer
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def forward(self, batch_dict: dict) -> dict:
        x = batch_dict['img']

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool is not None:
            x = self.maxpool(x)

        x = self.layer1(x)
        if self.save_intermediate:
            batch_dict['ResNet_intermediate_1'] = x
        x = self.layer2(x)
        if self.save_intermediate:
            batch_dict['ResNet_intermediate_2'] = x
        x = self.layer3(x)
        if self.save_intermediate:
            batch_dict['ResNet_intermediate_3'] = x
        x = self.layer4(x)
        if self.save_intermediate:
            batch_dict['ResNet_intermediate_4'] = x

        batch_dict['spatial_features'] = x
        return batch_dict

    def load_pretrained_weights(self):
        import torchvision
        
        weight_name = self.name + '_Weights.DEFAULT'
        model = getattr(torchvision.models, self.name.lower())(weights=weight_name)
        new_weights = get_updated_weights(self.state_dict(), model.state_dict())

        # conv1 shape is different in the pretrained model if init_downsample is False
        self.load_state_dict(new_weights, strict=self.init_downsample)
    
    @staticmethod
    def get_resnet_params(num_layers):
        resnet_name = 'ResNet{}'.format(num_layers)
        if num_layers == 18:
            return {'BLOCK': BasicBlock,
                    'NUM_BLOCKS': [2, 2, 2, 2]}, resnet_name
        elif num_layers == 34:
            return {'BLOCK': BasicBlock,
                    'NUM_BLOCKS': [3, 4, 6, 3]}, resnet_name
        elif num_layers == 50:
            return {'BLOCK': Bottleneck,
                    'NUM_BLOCKS': [3, 4, 6, 3]}, resnet_name
        elif num_layers == 101:
            return {'BLOCK': Bottleneck,
                    'NUM_BLOCKS': [3, 4, 23, 3]}, resnet_name
        elif num_layers == 152:
            return {'BLOCK': Bottleneck,
                    'NUM_BLOCKS': [3, 8, 36, 3]}, resnet_name
        else:
            raise ValueError("Invalid ResNet name: {}".format(resnet_name))
