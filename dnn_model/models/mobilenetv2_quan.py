import torch
import torch.nn as nn
import torch.nn.functional as F
from .quan_ops import conv2d_quantize_fn, activation_quantize_fn, batchnorm_fn

__all__ = ['mobilenetv2q']

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

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, acti_bit_list, weight_bit_list, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.acti_bit_list = acti_bit_list
        self.weight_bit_list = weight_bit_list
        self.wbit = self.weight_bit_list[-1]
        self.abit = self.acti_bit_list[-1]

        Conv2d = conv2d_quantize_fn(self.weight_bit_list)
        NormLayer = batchnorm_fn(self.weight_bit_list)

        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = NormLayer(planes)
        self.act1 = Activate(self.acti_bit_list)
        self.conv2 = Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = NormLayer(planes)
        self.act2 = Activate(self.acti_bit_list)
        self.conv3 = Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = NormLayer(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                NormLayer(out_planes),
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.act2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 2),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, acti_bit_list, weight_bit_list, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10

        self.acti_bit_list = acti_bit_list
        self.weight_bit_list = weight_bit_list
        self.wbit = self.weight_bit_list[-1]
        self.abit = self.acti_bit_list[-1]

        Conv2d = conv2d_quantize_fn(self.weight_bit_list)
        NormLayer = batchnorm_fn(self.weight_bit_list)

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = NormLayer(32)
        self.act1 = Activate(self.acti_bit_list)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = NormLayer(1280)
        self.act2 = Activate(self.acti_bit_list)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(self.acti_bit_list, self.weight_bit_list, in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = self.act2(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# For CIFAR10
def mobilenetv2q(acti_bit_list, weight_bit_list, num_classes=10, expand=5):
    return MobileNetV2(acti_bit_list, weight_bit_list, num_classes)