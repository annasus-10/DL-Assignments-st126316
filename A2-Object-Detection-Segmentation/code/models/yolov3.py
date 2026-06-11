import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'utils'))

import torch
import torch.nn as nn
import numpy as np
from utils import darknet
from utils.util import predict_transform


class MyDarknet(nn.Module):
    def __init__(self, cfgfile):
        super(MyDarknet, self).__init__()
        self.blocks = darknet.parse_cfg(cfgfile)
        self.net_info, self.module_list = darknet.create_modules(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        outputs = {}
        write = 0

        for i, module in enumerate(modules):
            module_type = (module["type"])

            if module_type == "convolutional" or module_type == "upsample":
                x = self.module_list[i](x)

            elif module_type == "route":
                layers = module["layers"]
                layers = [int(a) for a in layers]
                if (layers[0]) > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = outputs[i + (layers[0])]
                else:
                    if (layers[1]) > 0:
                        layers[1] = layers[1] - i
                    map1 = outputs[i + layers[0]]
                    map2 = outputs[i + layers[1]]
                    x = torch.cat((map1, map2), 1)

            elif module_type == "shortcut":
                from_ = int(module["from"])
                x = outputs[i-1] + outputs[i+from_]

            elif module_type == "yolo":
                anchors = self.module_list[i][0].anchors
                inp_dim = int(self.net_info["height"])
                num_classes = int(module["classes"])
                x = predict_transform(x, inp_dim, anchors, num_classes, CUDA)
                if not write:
                    detections = x
                    write = 1
                else:
                    detections = torch.cat((detections, x), 1)

            outputs[i] = x

        return detections

    def load_weights(self, weightfile):
        fp = open(weightfile, "rb")
        header = np.fromfile(fp, dtype=np.int32, count=5)
        self.header = torch.from_numpy(header)
        self.seen = self.header[3]
        weights = np.fromfile(fp, dtype=np.float32)
        fp.close()

        ptr = 0
        for i in range(len(self.module_list)):
            module_type = self.blocks[i + 1]["type"]
            if module_type != "convolutional":
                continue

            try:
                batch_normalize = int(self.blocks[i+1]["batch_normalize"])
            except:
                batch_normalize = 0

            model = self.module_list[i]
            conv  = model[0]

            if batch_normalize:
                bn = model[1]
                num_bn_biases = bn.bias.numel()

                bn_biases      = torch.from_numpy(weights[ptr:ptr+num_bn_biases]); ptr += num_bn_biases
                bn_weights     = torch.from_numpy(weights[ptr:ptr+num_bn_biases]); ptr += num_bn_biases
                bn_running_mean= torch.from_numpy(weights[ptr:ptr+num_bn_biases]); ptr += num_bn_biases
                bn_running_var = torch.from_numpy(weights[ptr:ptr+num_bn_biases]); ptr += num_bn_biases

                bn.bias.data.copy_(bn_biases.view_as(bn.bias.data))
                bn.weight.data.copy_(bn_weights.view_as(bn.weight.data))
                bn.running_mean.copy_(bn_running_mean.view_as(bn.running_mean))
                bn.running_var.copy_(bn_running_var.view_as(bn.running_var))
            else:
                num_biases = conv.bias.numel()
                conv_biases = torch.from_numpy(weights[ptr:ptr+num_biases]); ptr += num_biases
                conv.bias.data.copy_(conv_biases.view_as(conv.bias.data))

            num_weights = conv.weight.numel()
            conv_weights = torch.from_numpy(weights[ptr:ptr+num_weights]); ptr += num_weights
            conv.weight.data.copy_(conv_weights.view_as(conv.weight.data))