import argparse
import os
import time
import socket
import logging
from datetime import datetime
from functools import partial

import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.autograd import Variable
from tensorboardX import SummaryWriter

import models
from models.losses import CrossEntropyLossSoft
from datasets.data import get_dataset, get_transform
from optimizer import get_optimizer_config, get_lr_scheduler
from utils import setup_logging, save_checkpoint
from utils import AverageMeter, accuracy

from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP

# To prevent training from crashing due to truncated images
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

parser = argparse.ArgumentParser(description='Training')
parser.add_argument('--results-dir', default='./results', help='results dir')
parser.add_argument('--dataset', default='imagenet', help='dataset name or folder')
parser.add_argument('--train_split', default='train', help='train split name')
parser.add_argument('--model', default='mobilenetv2q', help='model architecture')
parser.add_argument('--workers', default=64, type=int, help='number of data loading workers')
parser.add_argument('--epochs', default=90, type=int, help='number of epochs')
parser.add_argument('--start-epoch', default=0, type=int, help='manual epoch number')
parser.add_argument('--batch-size', default=256*4, type=int, help='mini-batch size')
parser.add_argument('--optimizer', default='sgd', help='optimizer function used')
parser.add_argument('--lr', default=0, type=float, help='initial learning rate')
parser.add_argument('--lr_decay', default='45,60,70', help='lr decay steps')
parser.add_argument('--weight-decay', default=1e-5, type=float, help='weight decay')
parser.add_argument('--print-freq', '-p', default=20, type=int, help='print frequency')
parser.add_argument('--pretrain', default='./_pretrained/model_best_sep_12.pth.tar', help='path to pretrained full-precision checkpoint')
parser.add_argument('--resume', default=None, help='path to latest checkpoint')
parser.add_argument('--weight_bit_width_list', default='32', help='bit width list')
parser.add_argument('--acti_bit_width_list', default='32', help='bit width list')
args = parser.parse_args([])

LEAD_GPU = 3
ALL_GPU = [LEAD_GPU]

lead_gpu = f"cuda:{LEAD_GPU}"
device = torch.device(lead_gpu)
torch.cuda.set_device(lead_gpu)
torch.backends.cudnn.benchmark = True

def main():
    global device
    hostname = socket.gethostname()
    log_directory = "./logs/{}_{}/".format(args.model, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    # check if the log_directory exists  if not create it
    if not os.path.exists(log_directory):
        os.makedirs(log_directory)
    setup_logging(os.path.join(log_directory, 'log_{}.txt'.format(hostname)))
    logging.info("running arguments: %s", args)

    train_transform = get_transform(args.dataset, 'train')
    train_data = get_dataset(args.dataset, args.train_split, train_transform)
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True)

    val_transform = get_transform(args.dataset, 'val')
    val_data = get_dataset(args.dataset, 'val', val_transform)
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=args.batch_size,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    bit_width_list = list(map(int, args.weight_bit_width_list.split(',')))
    bit_width_list.sort()
    acti_bit_width_list = list(map(int, args.acti_bit_width_list.split(',')))
    model = models.__dict__[args.model](acti_bit_width_list, bit_width_list, train_data.num_classes)
    if len(ALL_GPU) > 1:
        model = nn.DataParallel(model, device_ids=ALL_GPU)
    model = model.to(device)
    model = model.half()


    lr_decay = list(map(int, args.lr_decay.split(',')))
    optimizer = get_optimizer_config(model, args.optimizer, args.lr, args.weight_decay)
    lr_scheduler = None
    best_prec1 = None
    if args.resume and args.resume != 'None':
        if os.path.isdir(args.resume):
            args.resume = os.path.join(args.resume, 'model_best.pth.tar')
        if os.path.isfile(args.resume):
            checkpoint = torch.load(args.resume, map_location=lead_gpu)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay, checkpoint['epoch'])
            logging.info("loaded resume checkpoint '%s' (epoch %s)", args.resume, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    elif args.pretrain and args.pretrain != 'None':
        if os.path.isdir(args.pretrain):
            args.pretrain = os.path.join(args.pretrain, 'model_best.pth.tar')
        if os.path.isfile(args.pretrain):
            checkpoint = torch.load(args.pretrain, map_location=lead_gpu)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            logging.info("loaded pretrain checkpoint '%s' (epoch %s)", args.pretrain, checkpoint['epoch'])
        else:
            raise ValueError('Pretrained model path error!')
    if lr_scheduler is None:
        lr_scheduler = get_lr_scheduler(args.optimizer, optimizer, lr_decay)
    num_parameters = sum([l.nelement() for l in model.parameters()])
    logging.info("number of parameters: %d", num_parameters)

    criterion = nn.CrossEntropyLoss().to(device)
    criterion_soft = CrossEntropyLossSoft().to(device)
    sum_writer = SummaryWriter(args.results_dir + '/summary')

    for epoch in range(args.start_epoch, args.epochs):
        model.train()
        train_loss, train_prec1, train_prec5 = forward(train_loader, model, criterion, criterion_soft, epoch, True,
                                                       optimizer, sum_writer)
        model.eval()
        val_loss, val_prec1, val_prec5 = forward(val_loader, model, criterion, criterion_soft, epoch, False)

        if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            lr_scheduler.step(val_loss)
        else:
            lr_scheduler.step()

        if best_prec1 is None:
            is_best = True
            best_prec1 = val_prec1[-1]
        else:
            is_best = val_prec1[-1] > best_prec1
            best_prec1 = max(val_prec1[-1], best_prec1)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'model': args.model,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer': optimizer.state_dict()
            },
            is_best,
            path=args.results_dir + '/ckpt')

        if sum_writer is not None:
            sum_writer.add_scalar('lr', optimizer.param_groups[0]['lr'], global_step=epoch)
            for bw, tl, tp1, tp5, vl, vp1, vp5 in zip(bit_width_list, train_loss, train_prec1, train_prec5, val_loss,
                                                      val_prec1, val_prec5):
                sum_writer.add_scalar('train_loss_{}'.format(bw), tl, global_step=epoch)
                sum_writer.add_scalar('train_prec_1_{}'.format(bw), tp1, global_step=epoch)
                sum_writer.add_scalar('train_prec_5_{}'.format(bw), tp5, global_step=epoch)
                sum_writer.add_scalar('val_loss_{}'.format(bw), vl, global_step=epoch)
                sum_writer.add_scalar('val_prec_1_{}'.format(bw), vp1, global_step=epoch)
                sum_writer.add_scalar('val_prec_5_{}'.format(bw), vp5, global_step=epoch)
        logging.info('Epoch {}: \ntrain loss {:.2f}, train prec1 {:.2f}, train prec5 {:.2f}\n'
                     '  val loss {:.2f},   val prec1 {:.2f},   val prec5 {:.2f}'.format(
                         epoch, train_loss[-1], train_prec1[-1], train_prec5[-1], val_loss[-1], val_prec1[-1],
                         val_prec5[-1]))


def forward(data_loader, model, criterion, criterion_soft, epoch, training=True, optimizer=None, sum_writer=None):
    global device
    bit_width_list = list(map(int, args.weight_bit_width_list.split(',')))
    bit_width_list.sort()
    losses = [AverageMeter() for _ in bit_width_list]
    top1 = [AverageMeter() for _ in bit_width_list]
    top5 = [AverageMeter() for _ in bit_width_list]

    tqdm_loader = tqdm(data_loader, desc='Epoch {}'.format(epoch))

    # for i, (input, target) in enumerate(data_loader):
    for i, (input, target) in enumerate(tqdm_loader):
        if not training:
            with torch.no_grad():
                # input = input.to(device)
                input = input.cuda(non_blocking=True)
                input = input.half()
                # target = target.to(device, non_blocking=True)
                target = target.cuda(non_blocking=True)

                for bw, am_l, am_t1, am_t5 in zip(bit_width_list, losses, top1, top5):
                    model.apply(lambda m: setattr(m, 'wbit', bw))
                    model.apply(lambda m: setattr(m, 'abit', bw))
                    output = model(input)
                    loss = criterion(output, target)

                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                    am_l.update(loss.item(), input.size(0))
                    am_t1.update(prec1.item(), input.size(0))
                    am_t5.update(prec5.item(), input.size(0))
        else:
            # input = input.to(device)
            input = input.cuda(non_blocking=True)
            input = input.half()
            # target = target.to(device, non_blocking=True)
            target = target.cuda(non_blocking=True)

            optimizer.zero_grad()

            # train full-precision supervisor
            model.apply(lambda m: setattr(m, 'wbit', bit_width_list[-1]))
            model.apply(lambda m: setattr(m, 'abit', bit_width_list[-1]))
            output = model(input)
            loss = criterion(output, target)
            loss.backward()
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses[-1].update(loss.item(), input.size(0))
            top1[-1].update(prec1.item(), input.size(0))
            top5[-1].update(prec5.item(), input.size(0))

            # train less-bit-wdith models
            target_soft = torch.nn.functional.softmax(output.detach(), dim=1)
            for j, (bw, am_l, am_t1, am_t5) in enumerate(
                    zip(bit_width_list[:-1][::-1], losses[:-1][::-1], top1[:-1][::-1], top5[:-1][::-1])):
                model.apply(lambda m: setattr(m, 'wbit', bw))
                model.apply(lambda m: setattr(m, 'abit', bw))
                output = model(input)
                # hard cross entropy
                # loss = criterion(output, target)
                # soft cross entropy
                loss = criterion_soft(output, target_soft)
                loss.backward()
                # recursive supervision
                target_soft = torch.nn.functional.softmax(output.detach(), dim=1)

                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                am_l.update(loss.item(), input.size(0))
                am_t1.update(prec1.item(), input.size(0))
                am_t5.update(prec5.item(), input.size(0))
            optimizer.step()

            tqdm_loader.set_postfix({
                'loss': losses[-1].val,
                'prec1': top1[-1].val,
                'prec5': top5[-1].val
            })
    return [_.avg for _ in losses], [_.avg for _ in top1], [_.avg for _ in top5]


if __name__ == '__main__':
    main()
