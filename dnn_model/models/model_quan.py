import torch
import torch.nn as nn
import torch.nn.functional as F

from .vgg import vgg16_bn
from .quan_ops import *

__all__ = ['vgg16q']

class Activate(nn.Module):
    def __init__(self, acti_bit_list, quantize=True):
        super(Activate, self).__init__()
        self.acti_bit_list = acti_bit_list
        self.abit = self.acti_bit_list[-1]
        self.acti = nn.ReLU(inplace=True)
        self.quantize = quantize
        if self.quantize:
            self.quan = activation_quantize_fn(self.acti_bit_list)

    def forward(self, x):
        if self.abit == 32:
            x = self.acti(x)
        else:
            x = torch.clamp(x, 0.0, 1.0)
        if self.quantize:
            x = self.quan(x)
        return x

class QATModule(nn.Module):
    def __init__(self, model: nn.Module, wbit_list, abit_list):
        super().__init__()
        self.model = model
        self.wbit_list = wbit_list
        self.abit_list = abit_list
        self.first_conv = True

        self.quant_model(self.model, self.wbit_list, self.abit_list)

    def quant_model(self, model, wbit_list, abit_list):
        
        for name, child_module in model.named_children():
            if isinstance(child_module, (nn.Conv2d)):
                if self.first_conv==True:
                    self.first_conv=False
                    continue
                Conv2d = conv2d_quantize_fn(wbit_list)
                original = model._modules[name]
                # bias = True if original.bias is not None else False

                model._modules[name] = Conv2d(
                                                in_channels=original.in_channels, 
                                                out_channels=original.out_channels, 
                                                kernel_size=original.kernel_size, 
                                                stride=original.stride, 
                                                padding=original.padding, 
                                                bias=False
                                                )
                model._modules[name].weight.data.copy_(original.weight.data)
                # if original.bias is not None:
                #     model._modules[name].bias.data.copy_(original.bias.data)
                
            elif isinstance(child_module, (nn.BatchNorm2d)):
                NormLayer = batchnorm_fn(wbit_list)
                original = model._modules[name]
                model._modules[name] = NormLayer(original.num_features)

                # # Copy weights and running stats
                # model._modules[name]._modules['bn_dict']._modules['32'].weight.data.copy_(original.weight.data)
                # model._modules[name]._modules['bn_dict']._modules['32'].bias.data.copy_(original.bias.data)
                # model._modules[name]._modules['bn_dict']._modules['32'].running_mean.data.copy_(original.running_mean.data)
                # model._modules[name]._modules['bn_dict']._modules['32'].running_var.data.copy_(original.running_var.data)
                # model._modules[name]._modules['bn_dict']._modules['32'].num_batches_tracked.data.copy_(original.num_batches_tracked.data)

            elif isinstance(child_module, (nn.ReLU)):
                model._modules[name] = Activate(wbit_list)
            else:
                self.quant_model(child_module, wbit_list, abit_list)

    def forward(self, input):
        return self.model(input)


def vgg16q(wbit_list, abit_list, num_classes):
    model = vgg16_bn(num_classes=num_classes, pretrained=False)
    quant_model = QATModule(model = model, wbit_list = wbit_list, abit_list = abit_list)
    return quant_model