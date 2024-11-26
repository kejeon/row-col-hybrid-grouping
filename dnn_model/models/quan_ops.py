# Adapted from
# https://github.com/zzzxxxttt/pytorch_DoReFaNet/blob/master/utils/quant_dorefa.py and
# https://github.com/tensorpack/tensorpack/blob/master/examples/DoReFa-Net/dorefa.py#L25

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SwitchBatchNorm2d(nn.Module):
    """Adapted from https://github.com/JiahuiYu/slimmable_networks
    """
    def __init__(self, num_features, weight_bit_list):
        super(SwitchBatchNorm2d, self).__init__()
        ''''
        4번 activation bit
        '''
        self.weight_bit_list = weight_bit_list
        self.bn_dict = nn.ModuleDict()

        for i in self.weight_bit_list:
            self.bn_dict[str(i)] = nn.BatchNorm2d(num_features)

        # if self.abit != self.wbit:
        #     raise ValueError('Currenty only support same activation and weight bit width!')
        self.wbit = self.weight_bit_list[-1]

    def forward(self, x):
        x = self.bn_dict[str(self.wbit)](x)
        return x


def batchnorm2d_fn(weight_bit_list):
    class SwitchBatchNorm2d_(SwitchBatchNorm2d):
        def __init__(self, num_features, weight_bit_list=weight_bit_list):
            super(SwitchBatchNorm2d_, self).__init__(num_features=num_features, weight_bit_list=weight_bit_list)

    return SwitchBatchNorm2d_



class qfn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, k): # k != bit, k == q_level
        n = float(k - 1)    

        out = torch.round(input * n) / n
        return out

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

# class qfn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input, k):
#         n = float(2**k - 1)
#         out = torch.round(input * n) / n
#         return out

#     @staticmethod
#     def backward(ctx, grad_output):
#         grad_input = grad_output.clone()
#         return grad_input, None

class qfn_acti(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, bit): # k != bit, k == q_level
        n = float(2**bit - 1)   

        out = torch.round(input * n) / n
        return out
    

    @staticmethod
    def backward(ctx, grad_output):
        grad_input = grad_output.clone()
        return grad_input, None

class weight_quantize_fn(nn.Module):
    def __init__(self, num_lvl_list):
        super(weight_quantize_fn, self).__init__()
        self.num_lvl_list = num_lvl_list
        self.num_lvl = self.num_lvl_list[-1]

    def forward(self, x):
        if self.num_lvl == 0:
            E = torch.mean(torch.abs(x)).detach()
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q_int = weight
            weight_q = weight * E
        else:
            E = torch.mean(torch.abs(x)).detach()
            quant_levels = float(self.num_lvl*2 -1)
            weight = torch.tanh(x)
            weight = weight / torch.max(torch.abs(weight))
            weight_q_int = qfn.apply(weight, self.num_lvl)
            weight_q = weight_q_int * E
            weight_q_int = weight_q_int*(self.num_lvl - 1)
        return weight_q, weight_q_int, E/(self.num_lvl - 1)

# class weight_quantize_fn(nn.Module):
#     def __init__(self, bit_list):
#         super(weight_quantize_fn, self).__init__()
#         self.bit_list = bit_list
#         self.wbit = self.bit_list[-1]
#         assert self.wbit <= 8 or self.wbit == 32

#     def forward(self, x):
#         if self.wbit == 32:
#             E = torch.mean(torch.abs(x)).detach()
#             weight = torch.tanh(x)
#             weight = weight / torch.max(torch.abs(weight))
#             weight_q = weight * E
#             # print(x.device)
#             # print(weight.device)
#             # print(weight_q.device)
#         else:
#             E = torch.mean(torch.abs(x)).detach()
#             weight = torch.tanh(x)
#             weight = weight / 2 / torch.max(torch.abs(weight)) + 0.5
#             weight_q = 2 * qfn.apply(weight, self.wbit) - 1
#             weight_q = weight_q * E
#         return weight_q, weight_q, E


class activation_quantize_fn(nn.Module):
    def __init__(self, acti_bit_list):
        super(activation_quantize_fn, self).__init__()
        self.abit = acti_bit_list[-1]
        # self.abit = self.bit_list[-1]
        assert self.abit <= 8 or self.abit == 32

    def forward(self, x):
        if self.abit == 32:
            activation_q = x
        else:
            activation_q = qfn_acti.apply(x, self.abit)
        return activation_q


# class Conv2d_Q(nn.Conv2d):
#     def __init__(self, k, in_planes, planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=False, mode = 'test'):
#         super(Conv2d_Q, self).__init__(in_planes, planes, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
#         self.k = k
#         self.quantize_fn = weight_quantize_fn(self.k)
#         # fp weight given. warm start through post training quantization
#         # 1. initialize with the inputted fp weight
#         # self.scaling_factor = 0
#         # self.w_q = self.weight
#         # self.w_q_int = self.weight

#         self.w_q, self.w_q_int, self.scaling_factor = self.quantize_fn(self.weight)
#         self.mode = mode

#     def forward(self, input):
#         # self.w_q = nn.Parameter(self.quantize_fn(self.weight)[0])
#         # self.w_q_int = nn.Parameter(self.quantize_fn(self.weight)[1])
#         # self.scaling_factor = nn.Parameter(self.quantize_fn(self.weight)[2])

#         self.w_q, self.w_q_int, self.scaling_factor = self.quantize_fn(self.weight)

#         # if self.mode == 'test':
#         #     mask = torch.ones_like(self.w_q)
#         #     self.w_q = self.w_q * mask

#         return F.conv2d(input, self.w_q, self.bias, self.stride, self.padding, self.dilation, self.groups)   # output: tensor

class Linear_Q(nn.Linear):
    def __init__(self, k, in_features, out_features, bias=True, requires_quant=True):
        super(Linear_Q, self).__init__(in_features, out_features, bias=bias)
        self.k = k
        self.w_quantize_fn = weight_quantize_fn(self.k)                         # weight qnt function
        self.b_quantize_fn = weight_quantize_fn(self.k)                         # bias qnt function
        self.requires_quant = requires_quant
        # fp weight given. warm start through post training quantization
        # 1. initialize with the inputted fp weight
        # self.scaling_factor = 0
        # self.w_q = self.weight
        # self.w_q_int = self.weight

        self.w_q, self.w_q_int, self.scaling_factor = self.w_quantize_fn(self.weight)
        self.b_q, self.b_q_int, _ = self.b_quantize_fn(self.bias) # 확인 필요

    def forward(self, input, order=None):
        # self.w_q = nn.Parameter(self.quantize_fn(self.weight)[0])
        # self.w_q_int = nn.Parameter(self.quantize_fn(self.weight)[1])
        # self.scaling_factor = nn.Parameter(self.quantize_fn(self.weight)[2])
        if self.requires_quant:
            self.w_q, self.w_q_int, self.scaling_factor = self.w_quantize_fn(self.weight)
            self.b_q, self.b_q_int, _ = self.b_quantize_fn(self.bias)
        
        return F.linear(input, self.w_q, self.b_q)


class Conv2d_Q(nn.Conv2d):
    def __init__(self, *kargs, **kwargs):
        super(Conv2d_Q, self).__init__(*kargs, **kwargs)


# def conv2d_quantize_fn(bit_list):
#     class Conv2d_Q_(Conv2d_Q):
#         def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
#                      bias=True):
#             super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
#                                             bias)
#             self.bit_list = bit_list
#             self.w_bit = self.bit_list[-1]
#             self.quantize_fn = weight_quantize_fn(self.bit_list)

#         def forward(self, input, order=None):
#             weight_q, _, _ = self.quantize_fn(self.weight)
#             return F.conv2d(input, weight_q, self.bias, self.stride, self.padding, self.dilation, self.groups)

#     return Conv2d_Q_

def conv2d_quantize_fn(k):
    class Conv2d_Q_(Conv2d_Q):
        def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1,
                     bias=True, requires_quant=True):
            super(Conv2d_Q_, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups,
                                            bias)
            self.k_list = k
            self.k = self.k_list[-1]
            self.quantize_fn = weight_quantize_fn(self.k_list)

            self.requires_quant = requires_quant
            self._forward = self._forward_with_quant if self.requires_quant else self._forward_without_quant

        def set_requires_quant(self, requires_quant):
            self.requires_quant = requires_quant
            self._forward = self._forward_with_quant if self.requires_quant else self._forward_without_quant

        def quantize_weight(self):
            self.w_q, self.w_q_int, self.scaling_factor = self.quantize_fn(self.weight)

        def _forward_without_quant(self, input):
            return F.conv2d(input, self.w_q, self.bias, self.stride, self.padding, self.dilation, self.groups) 
        
        def _forward_with_quant(self, input):
            self.w_q, _, _ = self.quantize_fn(self.weight)
            return F.conv2d(input, self.w_q, self.bias, self.stride, self.padding, self.dilation, self.groups) 
        
        def forward(self, input):
            return self._forward(input)

    return Conv2d_Q_


batchnorm_fn = batchnorm2d_fn

if __name__ == '__main__':
    # generate random torch tensor
    x = torch.rand(10)
    wqf = weight_quantize_fn(0)

    # quantize
    x_q, x_q_int, S = wqf(torch.tensor(-2))

    # print
    print(x)
    print(x_q)

