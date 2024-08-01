import torch.nn as nn

from .backbone_template import BackboneTemplate


class ConvNetBackbone(BackboneTemplate):
    def __init__(self, backbone_config):
        super(ConvNetBackbone, self).__init__(backbone_config)
        self.input_key = backbone_config.get('INPUT_KEY', 'image')
        self.output_key = backbone_config.get('OUTPUT_KEY', 'spatial_features')

        use_bn = backbone_config.get('USE_BN', False)
        last_bn = backbone_config.get('LAST_BN', False)
        use_relu = backbone_config.get('USE_RELU', False)
        last_relu = backbone_config.get('LAST_RELU', False)

        module_list = []
        modules_config = backbone_config.MODULES
        for i, module_config in enumerate(modules_config[:-1]):
            params = module_config.get('PARAMS', {})

            module = [getattr(nn, module_config.NAME)(**params)]
            if module_config.NAME == 'Conv2d':
                if use_bn:
                    module.append(nn.BatchNorm2d(module.out_channels))
                if use_relu:
                    module.append(nn.ReLU(inplace=True))

            if len(module) > 1:
                module = nn.Sequential(*module)
            else:
                module = module[0]
            module_list.append(module)


        last_module = getattr(nn, modules_config[-1].NAME)(**modules_config[-1].PARAMS)
        module_list.append(last_module)

        if last_bn:
            module_list.append(nn.BatchNorm2d(last_module.out_channels))
        if last_relu:
            module_list.append(nn.ReLU(inplace=True))

        self.backbone = nn.Sequential(*module_list)

        self.code = backbone_config.get('CODE', 'ConvNet')
        self.save_intermediate_features = backbone_config.get('SAVE_INTERMEDIATE_FEATURES', [])
        

    def forward(self, batch_dict):
        x = batch_dict[self.input_key]
        for i, m in enumerate(self.backbone):
            x = m(x)
            if i in self.save_intermediate_features:
                batch_dict[f'{self.code}_intermediate_{i}'] = x
        batch_dict[self.output_key] = x
        return batch_dict
