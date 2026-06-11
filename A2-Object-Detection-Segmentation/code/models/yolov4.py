import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import torch
import torch.nn as nn
import numpy as np
from utils import darknet
from utils.util import predict_transform


class Mish(nn.Module):
    """Mish activation: x * tanh(softplus(x))
    Used in YOLOv4's CSPDarknet53 backbone instead of LeakyReLU.
    """
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


def create_modules_v4(blocks):
    """Extended create_modules that adds Mish activation and maxpool support for YOLOv4."""
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    output_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x['type'] == 'convolutional':
            activation = x['activation']
            try:
                batch_normalize = int(x['batch_normalize'])
                bias = False
            except:
                batch_normalize = 0
                bias = True

            filters     = int(x['filters'])
            padding     = int(x['pad'])
            kernel_size = int(x['size'])
            stride      = int(x['stride'])
            pad         = (kernel_size - 1) // 2 if padding else 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias=bias)
            module.add_module(f'conv_{index}', conv)

            if batch_normalize:
                module.add_module(f'batch_norm_{index}', nn.BatchNorm2d(filters))

            if activation == 'leaky':
                module.add_module(f'leaky_{index}', nn.LeakyReLU(0.1, inplace=True))
            elif activation == 'mish':
                module.add_module(f'mish_{index}', Mish())

        elif x['type'] == 'maxpool':
            size   = int(x['size'])
            stride = int(x.get('stride', 1))
            # YOLOv4 uses maxpool with stride=1 and same padding in SPP block
            if stride == 1:
                pad = (size - 1) // 2
                module.add_module(f'maxpool_{index}',
                    nn.MaxPool2d(size, stride=stride, padding=pad))
            else:
                module.add_module(f'maxpool_{index}',
                    nn.MaxPool2d(size, stride=stride))
            filters = prev_filters

        elif x['type'] == 'upsample':
            module.add_module(f'upsample_{index}',
                nn.Upsample(scale_factor=int(x['stride']), mode='nearest'))
            filters = prev_filters

        elif x['type'] == 'route':
            # YOLOv4 route can concatenate more than 2 layers
            layer_indices = [int(a) for a in x['layers'].split(',')]
            filters = 0
            for l in layer_indices:
                if l > 0:
                    filters += output_filters[l]
                else:
                    filters += output_filters[index + l]
            module.add_module(f'route_{index}', darknet.EmptyLayer())

        elif x['type'] == 'shortcut':
            module.add_module(f'shortcut_{index}', darknet.EmptyLayer())
            filters = prev_filters

        elif x['type'] == 'yolo':
            mask    = [int(m) for m in x['mask'].split(',')]
            anchors = [int(a) for a in x['anchors'].split(',')]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in mask]
            module.add_module(f'Detection_{index}',
                darknet.DetectionLayer(anchors))

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)

    return net_info, module_list


class MyDarknetV4(nn.Module):
    """YOLOv4 — extends MyDarknet with Mish, maxpool, and 3+ layer route support.
    Input: 608x608 RGB (not BGR).
    """
    def __init__(self, cfgfile):
        super(MyDarknetV4, self).__init__()
        self.blocks = darknet.parse_cfg(cfgfile)
        self.net_info, self.module_list = create_modules_v4(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for i, module in enumerate(modules):
            module_type = module['type']

            if module_type in ('convolutional', 'upsample', 'maxpool'):
                x = self.module_list[i](x)

            elif module_type == 'route':
                # Supports 2+ layers (YOLOv4 uses up to 4)
                layer_indices = [int(a) for a in module['layers'].split(',')]
                maps = []
                for l in layer_indices:
                    if l > 0:
                        maps.append(outputs[l])
                    else:
                        maps.append(outputs[i + l])
                x = torch.cat(maps, dim=1)

            elif module_type == 'shortcut':
                from_ = int(module['from'])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == 'yolo':
                anchors     = self.module_list[i][0].anchors
                inp_dim     = int(self.net_info['height'])
                num_classes = int(module['classes'])
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, 'rb')
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen   = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i+1]['type']
            if module_type != 'convolutional':
                continue

            try:
                batch_normalize = int(self.blocks[i+1]['batch_normalize'])
            except:
                batch_normalize = 0

            model = self.module_list[i]
            conv  = model[0]

            if batch_normalize:
                bn = model[1]
                num_bn_biases = bn.bias.numel()

                bn_biases       = torch.from_numpy(weights[ptr:ptr+num_bn_biases]); ptr += num_bn_biases
                bn_weights      = torch.from_numpy(weights[ptr:ptr+num_bn_biases]); ptr += num_bn_biases
                bn_running_mean = torch.from_numpy(weights[ptr:ptr+num_bn_biases]); ptr += num_bn_biases
                bn_running_var  = torch.from_numpy(weights[ptr:ptr+num_bn_biases]); ptr += num_bn_biases

                bn.bias.data.copy_(bn_biases.view_as(bn.bias.data))
                bn.weight.data.copy_(bn_weights.view_as(bn.weight.data))
                bn.running_mean.copy_(bn_running_mean.view_as(bn.running_mean))
                bn.running_var.copy_(bn_running_var.view_as(bn.running_var))
            else:
                num_biases  = conv.bias.numel()
                conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases]); ptr += num_biases
                conv.bias.data.copy_(conv_biases.view_as(conv.bias.data))

            num_weights  = conv.weight.numel()
            conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights]); ptr += num_weights
            conv.weight.data.copy_(conv_weights.view_as(conv.weight.data))