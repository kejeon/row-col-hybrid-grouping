# Refer to https://arxiv.org/abs/1512.03385
import torch
import torch.nn as nn
import torch.nn.functional as F
from .quan_ops import conv2d_quantize_fn, activation_quantize_fn, batchnorm_fn

__all__ = ['resnet20q', 'resnet18q', 'resnet50q']


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


class PreActBasicBlockQ(nn.Module):
    """Pre-activation version of the BasicBlock.
    """
    def __init__(self, acti_bit_list, weight_bit_list, in_planes, out_planes, stride=1):
        super(PreActBasicBlockQ, self).__init__()
        # self.bit_list = bit_list
        self.acti_bit_list = acti_bit_list
        self.weight_bit_list = weight_bit_list
        self.wbit = self.weight_bit_list[-1]
        self.abit = self.acti_bit_list[-1]

        Conv2d = conv2d_quantize_fn(self.weight_bit_list)
        NormLayer = batchnorm_fn(self.weight_bit_list) # acti_bit_list

        self.bn0 = NormLayer(in_planes)
        self.act0 = Activate(self.acti_bit_list)
        self.conv0 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(out_planes)
        self.act1 = Activate(self.acti_bit_list)
        self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.skip_conv = None
        if stride != 1:
            self.skip_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)
            self.skip_bn = nn.BatchNorm2d(out_planes)

    def forward(self, x):
        out = self.bn0(x)
        out = self.act0(out)

        if self.skip_conv is not None:
            shortcut = self.skip_conv(out)
            shortcut = self.skip_bn(shortcut)
        else:
            shortcut = x

        out = self.conv0(out)
        out = self.bn1(out)
        out = self.act1(out)
        out = self.conv1(out)
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_units, acti_bit_list, weight_bit_list, num_classes, expand=5):
        super(PreActResNet, self).__init__()
        # self.bit_list = bit_list
        self.weight_bit_list = weight_bit_list
        self.acti_bit_list = acti_bit_list
        self.wbit = self.weight_bit_list[-1]
        self.abit = self.acti_bit_list[-1]
        self.expand = expand

        NormLayer = batchnorm_fn(self.weight_bit_list) # acti_bit_list

        ep = self.expand
        self.conv0 = nn.Conv2d(3, 16 * ep, kernel_size=3, stride=1, padding=1, bias=False)

        strides = [1] * num_units[0] + [2] + [1] * (num_units[1] - 1) + [2] + [1] * (num_units[2] - 1)
        channels = [16 * ep] * num_units[0] + [32 * ep] * num_units[1] + [64 * ep] * num_units[2]
        in_planes = 16 * ep
        self.layers = nn.ModuleList()
        for stride, channel in zip(strides, channels):
            self.layers.append(block(self.acti_bit_list, self.weight_bit_list, in_planes, channel, stride))
            in_planes = channel

        self.bn = NormLayer(64 * ep)
        self.fc = nn.Linear(64 * ep, num_classes)

    def forward(self, x):
        out = self.conv0(x)
        for layer in self.layers:
            out = layer(out)
        out = self.bn(out)
        out = out.mean(dim=2).mean(dim=2)
        out = self.fc(out)
        return out


class PreActBottleneckQ(nn.Module):
    expansion = 4

    def __init__(self, acti_bit_list, weight_bit_list, in_planes, out_planes, stride=1, downsample=None):
        super(PreActBottleneckQ, self).__init__()
        self.weight_bit_list = weight_bit_list
        self.acti_bit_list = acti_bit_list
        self.wbit = self.weight_bit_list[-1]
        self.abit = self.acti_bit_list[-1]

        Conv2d = conv2d_quantize_fn(self.weight_bit_list)
        norm_layer = batchnorm_fn(self.weight_bit_list)

        self.bn0 = norm_layer(in_planes)
        self.act0 = Activate(self.acti_bit_list)
        self.conv0 = Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)
        self.bn1 = norm_layer(out_planes)
        self.act1 = Activate(self.acti_bit_list)
        self.conv1 = Conv2d(out_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = norm_layer(out_planes)
        self.act2 = Activate(self.acti_bit_list)
        self.conv2 = Conv2d(out_planes, out_planes * self.expansion, kernel_size=1, stride=1, bias=False)
        self.downsample = downsample

    def forward(self, x):        
        shortcut = self.downsample(x) if self.downsample is not None else x
        out = self.conv0(self.act0(self.bn0(x)))
        out = self.conv1(self.act1(self.bn1(out)))
        out = self.conv2(self.act2(self.bn2(out)))
        out += shortcut
        return out


class PreActResNetBottleneck(nn.Module):
    def __init__(self, block, layers, acti_bit_list, weight_bit_list, num_classes):
        super(PreActResNetBottleneck, self).__init__()
        self.weight_bit_list = weight_bit_list
        self.acti_bit_list = acti_bit_list
        self.wbit = self.weight_bit_list[-1]
        self.abit = self.acti_bit_list[-1]

        self.norm_layer = batchnorm_fn(self.weight_bit_list)

        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.bn = self.norm_layer(512 * block.expansion)
        self.act = Activate(self.acti_bit_list)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                self.norm_layer(planes * block.expansion))

        layers = []
        layers.append(block(self.acti_bit_list, self.weight_bit_list, self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.acti_bit_list, self.weight_bit_list, self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.act(self.bn(x))
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, acti_bit_list, weight_bit_list, in_planes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, trunc=False):
        super(BasicBlock, self).__init__()

        self.acti_bit_list = acti_bit_list
        self.weight_bit_list = weight_bit_list
        self.wbit = self.weight_bit_list[-1]
        self.abit = self.acti_bit_list[-1]
    
        Conv2d = conv2d_quantize_fn(self.weight_bit_list)
        NormLayer = batchnorm_fn(self.weight_bit_list)

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        
        # self.bn1 = NormLayer(in_planes)
        # self.act1 = Activate(self.acti_bit_list)
        self.conv1 = Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = NormLayer(planes)
        self.act1 = Activate(self.acti_bit_list)

        self.conv2 = Conv2d(planes, planes, kernel_size=3, padding=1, stride=1, bias=False) 
        self.bn2 = NormLayer(planes)
        self.act2 = Activate(self.acti_bit_list)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act2(out)

        return out

# TODO: Outdated. Remove later
# class BasicBlock(nn.Module):
# 	# mul은 추후 ResNet18, 34, 50, 101, 152등 구조 생성에 사용됨
#     mul = 1
#     def __init__(self, acti_bit_list, weight_bit_list, in_planes, out_planes, stride=1):
#     # def __init__(self, in_planes, out_planes, stride=1):
#         super(BasicBlock, self).__init__()
        
#         self.acti_bit_list = acti_bit_list
#         self.weight_bit_list = weight_bit_list
#         self.wbit = self.weight_bit_list[-1]
#         self.abit = self.acti_bit_list[-1]

#         Conv2d = conv2d_quantize_fn(self.weight_bit_list)
#         NormLayer = batchnorm_fn(self.weight_bit_list)

#         self.bn1 = NormLayer(in_planes)
#         self.act1 = Activate(self.acti_bit_list)
#         self.conv1 = Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = NormLayer(out_planes)
#         self.act2 = Activate(self.acti_bit_list)
#         self.conv2 = Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)


#         # # stride를 통해 너비와 높이 조정
#         # self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         # self.bn1 = nn.BatchNorm2d(out_planes)
        
#         # # stride = 1, padding = 1이므로, 너비와 높이는 항시 유지됨
#         # self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
#         # self.bn2 = nn.BatchNorm2d(out_planes)
        
#         # # x를 그대로 더해주기 위함
#         # self.shortcut = nn.Sequential()
        
#         # 만약 size가 안맞아 합연산이 불가하다면, 연산 가능하도록 모양을 맞춰줌
#         if stride != 1: # x와 
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_planes)
#             )
    
#     def forward(self, x):
#         out = self.conv0(x)
#         out = self.bn1(out)
#         out = self.act1(out)
#         out = self.conv1(out)
#         out = self.bn2(out)
#         out += self.shortcut(x) # 필요에 따라 layer를 Skip
#         out = self.acti1(out)
#         return out

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResNet(nn.Module):

    def __init__(self, block, layers, acti_bit_list, weight_bit_list, 
                 num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, trunc=False):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.acti_bit_list = acti_bit_list
        self.weight_bit_list = weight_bit_list
        self.wbit = self.weight_bit_list[-1]
        self.abit = self.acti_bit_list[-1]
        self.inplanes = 64
        self.dilation = 1

        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], trunc=trunc)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0], trunc=trunc)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1], trunc=trunc)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2], trunc=trunc)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                # if isinstance(m, Bottleneck):
                #     nn.init.constant_(m.bn3.weight, 0)
                if isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False, trunc=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),)

        layers = []
        layers.append(block(self.acti_bit_list, self.weight_bit_list, self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, trunc))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.acti_bit_list, self.weight_bit_list, self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, trunc=trunc))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x):
        return self._forward_impl(x) 

# TODO: Outdated. Remove later
# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks, acti_bit_list, weight_bit_list, num_classes=1000):
    
#         super(ResNet, self).__init__()
#         self.in_planes = 64
#         self.acti_bit_list = acti_bit_list
#         self.weight_bit_list = weight_bit_list
#         self.wbit = self.weight_bit_list[-1]
#         self.abit = self.acti_bit_list[-1]
        
#         # Resnet 논문 구조의 conv1 파트 그대로 구현
#         self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7, stride=2, padding = 3)
#         self.bn1 = nn.BatchNorm2d(self.in_planes)
#         self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
#         self.layer1 = self.make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self.make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self.make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self.make_layer(block, 512, num_blocks[3], stride=2)
#         self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
#         # Basic Resiudal Block일 경우 그대로, BottleNeck일 경우 4를 곱한다.
#         self.linear = nn.Linear(512 * block.mul, num_classes)
        
#     # 다양한 Architecture 생성을 위해 make_layer로 Sequential 생성     
#     def make_layer(self, block, out_planes, num_blocks, stride):
#         # layer 앞부분에서만 크기를 절반으로 줄이므로, 아래와 같은 구조
#         strides = [stride] + [1] * (num_blocks-1)
#         layers = []
#         for i in range(num_blocks):
#             layers.append(block(self.acti_bit_list, self.weight_bit_list, self.in_planes, out_planes, strides[i]))
#             self.in_planes = block.expansion * out_planes
#         return nn.Sequential(*layers)
    
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = F.relu(out)
#         out = self.maxpool1(out)
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = torch.flatten(out,1)
#         out = self.linear(out)
#         return out


# For CIFAR10
def resnet20q(acti_bit_list, weight_bit_list, num_classes=10, expand=5):
    return PreActResNet(PreActBasicBlockQ, [3, 3, 3], acti_bit_list, weight_bit_list, num_classes=num_classes, expand=expand)

# For ImageNet
def resnet50q(acti_bit_list, weight_bit_list, num_classes=1000):
    return PreActResNetBottleneck(PreActBottleneckQ, [3, 4, 6, 3], acti_bit_list, weight_bit_list, num_classes=num_classes)

def resnet18q(acti_bit_list, weight_bit_list, num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], acti_bit_list, weight_bit_list, num_classes = num_classes)