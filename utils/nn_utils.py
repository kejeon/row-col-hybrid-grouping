import dnn_model.models as models
from dnn_model.datasets.data import get_dataset, get_transform
from dnn_model.utils import AverageMeter, accuracy
from dnn_model.models.quan_ops import Conv2d_Q

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data

import numpy as np
from functools import reduce, partial
from tqdm import tqdm
from .nn_data import dataset_model_dict, pretrained_paths, parallel_settings


def setup_gpus(lead_gpu_idx):
    best_gpu = f"cuda:{lead_gpu_idx}"
    device = torch.device(best_gpu)
    torch.cuda.set_device(best_gpu)
    torch.backends.cudnn.benchmark = True
    return device

def load_train_dataset(dataset='cifar10',
                        batch_size=1024,
                        num_workers=2):
    train_transform = get_transform(dataset, 'train')
    train_data = get_dataset(dataset, 'train', train_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                persistent_workers=True)
    return train_data, train_loader

def load_val_dataset(dataset='cifar10',
                     batch_size=1024,
                     num_workers=2):
    val_transform = get_transform(dataset, 'val')
    val_data = get_dataset(dataset, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_data,
                                                batch_size=batch_size,
                                                shuffle=False,
                                                num_workers=num_workers,
                                                pin_memory=True,
                                                persistent_workers=True)
    return val_data, val_loader

def get_dataset_for_model(model_name, 
                            split, 
                            batch_size, 
                            num_workers):
    # dataset_model_dict = {
    #     'resnet50q': 'imagenet',
    #     'resnet18q': 'imagenet',
    #     'resnet20q': 'cifar10',
    #     'resnet20qe1': 'cifar10',
    # }

    if model_name not in dataset_model_dict:
        raise ValueError(f"Model {model_name} not found.")

    dataset = dataset_model_dict[model_name]

    if split == 'train':
        data, loader = load_train_dataset(dataset=dataset, 
                                          batch_size=batch_size, 
                                          num_workers=num_workers)
    elif split == 'val':
        data, loader = load_val_dataset(dataset=dataset, 
                                        batch_size=batch_size, 
                                        num_workers=num_workers)
    else:
        raise ValueError(f"Split {split} not found.")
    
    return data, loader

def get_model(model_name, 
              acti_bit, 
              weight_bit, 
              num_classes, 
              all_gpu_idx, 
              device):
    if model_name not in pretrained_paths:
        raise ValueError(f"Model {model_name} not found.")
    
    if model_name not in models.__dict__:
        if model_name == 'resnet20qe1':
            model = models.__dict__['resnet20q'](acti_bit, weight_bit, num_classes=num_classes, expand=1)
    else:
        model = models.__dict__[model_name](acti_bit, weight_bit, num_classes=num_classes)
    parallel = parallel_settings[model_name]

    if parallel:
        model = nn.DataParallel(model, device_ids=all_gpu_idx)
    
    model = model.to(device)
    checkpoint = torch.load(pretrained_paths[model_name], map_location=device)
    missing, _ = model.load_state_dict(checkpoint['state_dict'], strict=False)

    if len(missing) > 0:
        raise ValueError(f"Missing keys: {missing}")

    return model

def get_layer_by_name(model, layer_name):
    return reduce(getattr, layer_name.split('.'), model)

def set_precision(model, prec_config):
    for layer_name, quant_prec in prec_config:
        nearest_bit_prec = int(np.log2(quant_prec)) + 1
        res_layer = get_layer_by_name(model, layer_name)
        res_layer.apply(lambda m: setattr(m, 'wbit', nearest_bit_prec))
        res_layer.apply(lambda m: setattr(m, 'num_lvl', quant_prec))

# Forward pass function
def forward(model, dataset_loader, device, criterion, half_prec, tqdm_desc=None):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    loss_acc = None

    loader_tqdm = tqdm(dataset_loader)
    for i, (input, target) in enumerate(loader_tqdm):
        if tqdm_desc is not None:
            loader_tqdm.set_description(tqdm_desc)
        input = input.to(device, non_blocking=True)
        if half_prec:
            input = input.half()
        target = target.to(device, non_blocking=True)

        output = model(input)
        loss = criterion(output, target)
        # loss.backward()
        # if loss_acc is None:
        #     loss_acc = output
        # else:
        #     loss_acc += loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

        losses.update(loss.item(), input.size(0))
        top1.update(prec1.item(), input.size(0))
        top5.update(prec5.item(), input.size(0))

        loader_tqdm.set_postfix({'top1': top1.avg, 'top5': top5.avg, 'loss': losses.avg})

    return top1.avg, top5.avg, losses.avg, loss

# Function for evaluation
def forward_evaluation(model, dataset_loader, device, criterion, half_prec):
    model.eval()
    with torch.no_grad():
        top1, top5, loss, _ = forward(model, dataset_loader, device, criterion, half_prec, tqdm_desc='Evaluation')
    return top1, top5, loss

# Adaptive batch normalization
def forward_adaptiveBN(model, dataset_loader, device, criterion, half_prec):
    model.train()
    with torch.no_grad():
        top1, top5, loss, _ = forward(model, dataset_loader, device, criterion, half_prec, tqdm_desc='Adaptive BN')
    return top1, top5, loss

def forward_gen_func(device, criterion, half_prec):
    forward_functions = [
        forward,
        forward_evaluation,
        forward_adaptiveBN
    ]
    partial_functions = {}
    
    for fn in forward_functions:
        partial_fn = partial(
            fn, 
            device=device, 
            criterion=criterion, 
            half_prec=half_prec)
        partial_functions[fn.__name__] = partial_fn

    return partial_functions

def update_requires_quant_conv2d(model, requires_quant):
    for _, layer in model.named_modules():
        if isinstance(layer, Conv2d_Q):
            layer.set_requires_quant(requires_quant)
    return 

def quantize_conv2d(model):
    for _, layer in model.named_modules():
        if isinstance(layer, Conv2d_Q):
            layer.quantize_weight()

def q_freeze_conv2d(model):
    update_requires_quant_conv2d(model, False)
    return

def q_unfreeze_conv2d(model):
    update_requires_quant_conv2d(model, True)
    return

def get_input_size_with_hooks(model, input_data):
    input_size_dict = {}
    def hook(module, input, output, layer_name):
        input_size = input[0].shape[1:]
        input_size_dict[layer_name] = tuple(input_size)

    hooks = []
    for name, layer in model.named_modules():
        if isinstance(layer, Conv2d_Q):
            hook_handle = layer.register_forward_hook(lambda m, i, o, n=name: hook(m, i, o, n))
            hooks.append(hook_handle)

    model(input_data)

    for hook in hooks:
        hook.remove()

    return input_size_dict


# TODO: Legacy functions, remove later
# def forward(model, dataloader, bit_width, criterion):
#     model.apply(lambda m: setattr(m, 'wbit', bit_width))
#     model.apply(lambda m: setattr(m, 'k', bit_width))

#     losses = AverageMeter()
#     top1 = AverageMeter()
#     top5 = AverageMeter()

#     for i, (input, target) in enumerate(dataloader):
#         with torch.no_grad():
#             input = input.cuda()
#             target = target.cuda()
#             output = model(input)
#             loss = criterion(output, target)
#             prec1, prec5 = accuracy(output.data, target, topk=(1, 5))

#             losses.update(loss.item(), input.size(0))
#             top1.update(prec1.item(), input.size(0))
#             top5.update(prec5.item(), input.size(0))
    
#     return losses.avg, top1.avg, top5.avg

# def forward_bitwise_gen_func(model, dataloader, criterion):
#     def forward_bitwise(bit_width):
#         return forward(model, dataloader, bit_width, criterion)
#     return forward_bitwise