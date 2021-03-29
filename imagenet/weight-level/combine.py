import argparse
import os
import random
import shutil
import time
import warnings
import numpy as np
import torchsnooper
import torchsummary

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#only used data to compute accuracy, not in deciding which to prune

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=90, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=64, type=int,
                    metavar='N', help='mini-batch size (default: 64)')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--print-freq', '-p', default=1, type=int,
                    metavar='N', help='print frequency (default: 1)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--world-size', default=1, type=int,
                    help='number of distributed processes')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='gloo', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')

best_prec1 = 0
mean = [ 0.485, 0.456, 0.406 ]
std = [ 0.229, 0.224, 0.225 ]

transform = transforms.Compose([
    transforms.Scale(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean = mean, std = std),
    ])

def default_loader(path):
    return Image.open(path).convert('RGB')

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, txt, transform=None, target_transform=None, loader=default_loader):
        fh = open(txt, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            line = line.strip('\n')
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0],int(words[1])))
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        fn, label = self.imgs[index]
        img = self.loader(fn)
        if self.transform is not None:
            img = self.transform(img)
        return img,label

    def __len__(self):
        return len(self.imgs)

#@torchsnooper.snoop()
def main():
    global args, best_prec1
    args = parser.parse_args()

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    args.distributed = args.world_size > 1

    if args.distributed:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size)

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True)
    else:

        print("=> creating model '{}'".format(args.arch))
        model_1 = models.__dict__[args.arch]()
        num_ftrs = model_1.classifier[6].in_features
        model_1.classifier[6] = nn.Linear(num_ftrs, 102) #only train the last layer

        model_2 = models.__dict__[args.arch]()
        num_ftrs = model_2.classifier[6].in_features
        model_2.classifier[6] = nn.Linear(num_ftrs, 102) #only train the last layer

        print("=> loading pre-trained model '{}'".format(args.arch))
        model_0 = models.__dict__[args.arch](pretrained=True)
        #model_0.classifier[6] = nn.Linear(num_ftrs, 102) #only train the last layer
       

    

    if args.gpu is not None:
        model_1 = model_1.cuda(args.gpu) #this way
        model_2 = model_2.cuda(args.gpu) #this way
        model_0 = model_0.cuda(args.gpu) #this way
    elif args.distributed:
        model_1.cuda()
        model_2.cuda()
        model_0.cuda()
        model_1 = torch.nn.parallel.DistributedDataParallel(model_1)
        model_2 = torch.nn.parallel.DistributedDataParallel(model_2)
        model_0 = torch.nn.parallel.DistributedDataParallel(model_0)

    else:
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model_1.features = torch.nn.DataParallel(model_1.features)
            model_2.features = torch.nn.DataParallel(model_2.features)
            model_0.features = torch.nn.DataParallel(model_0.features)
            model_1.cuda()
            model_2.cuda()
            model_0.cuda()

        else:
            model = torch.nn.DataParallel(model).cuda()
    
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint ..')

        assert os.path.isfile("/root/hcc/exp1/save/wl/prune/scratch_vgg9.pth.tar"), 'Error: no checkpoint1-scratch directory found!'
        checkpoint1 = torch.load("/root/hcc/exp1/save/wl/prune/scratch_vgg9.pth.tar").get('state_dict')

        model_1.load_state_dict(checkpoint1) #cat dog        print('==> Resuming from checkpoint ..')

        assert os.path.isfile("/root/hcc/exp1/save/wl/prune/transformed_vgg.pth.tar"), 'Error: no checkpoint2-transform directory found!'
        checkpoint2 = torch.load("/root/hcc/exp1/save/wl/prune/transformed_vgg.pth.tar").get('state_dict')

        model_2.load_state_dict(checkpoint2) #cat dog


    val_dataset = MyDataset(txt=args.data+'dataset-val2.txt', transform=transform)
    val_loader = torch.utils.data.DataLoader(val_dataset , batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)


    criterion = nn.CrossEntropyLoss().cuda(args.gpu)    
    test_acc = validate(val_loader, model_1,model_2, model_0, criterion)
   

    
    return

def validate(val_loader, model_1,model_2, model_0 , criterion):
    # AverageMeter() : Computes and stores the average and current value
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    #zero_tensor = torch.FloatTensor(n, m)

    # switch to evaluate mode
    model_1.eval()
    model_2.eval()
    model_0.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True) # 0*100 + 1*100 +2*100
            print("target:",target)

            # compute output,out put is a tensor
            output_1 = model_1(input)
            output_1= F.softmax(output_1, dim=1) # calculate as row
            #print("output_1:",output_1)
 
            output_0 = model_0(input)
            output_0= F.softmax(output_0, dim=1)
            #print("output_0:",output_0)

            output_2 = model_2(input)
            output_2= F.softmax(output_2, dim=1)
            #print(output_2)

            out_size = output_1.size()
            row = out_size[0] 
            zero_tensor = torch.FloatTensor(row,1).zero_().cuda()


            #print(output_2)

            #print("output_1:",output_1)
            #print("output_0:",output_0)

            loss = criterion(output_2, target)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output_1, output_2, target, topk=(1, 2))

            #print("-------------ImageNet------------------")
            #accuracy(output_0, target, topk=(1, 5))
            #print("-------------over------------------")

            #print("prec1:",prec1)
            #print("prec5:",prec5)

            losses.update(loss.item(), input.size(0))
            top1.update(prec1[0], input.size(0))
            top5.update(prec5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

def accuracy(output_1,output_2, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #view() means resize() -1 means 'it depends'
    with torch.no_grad():
        batch_size = target.size(0)
        #print("batch_size",batch_size)
        maxk = max(topk) # = 5
        number1, pred1 = output_1.topk(maxk, 1, True, True) #sort and get top k and their index
        pred = pred1[:]
        number2, pred2 = output_2.topk(maxk, 1, True, True) #sort and get top k and their index
        print("pred1:",pred1.t()) #is index 5col xrow
        print("pred2:",pred2.t()) #is index 5col xrow
        #print("pred after:",pred)
        print("_ number1 after:",number1.t())

        #print("number1[0][1]:",number1[0][1])
        #print("pred[0][1]:",pred[0][1])
        print("number1.shape[0]",number1.shape[0])
        print("number1.shape[1]",number1.shape[1])
        for a in range(0,number1.shape[0]):
            gap_0 = number1[a][0] - number1[a][1]
            gap = gap_0[0]
            print("gap:",gap)
            print("a:",a)
            if gap < 0.6:
                pred[a][0] = pred2[a][0]
                pred[a][1] = pred2[a][1]
                
            

        pred = pred.t() # a zhuanzhi transpose xcol 5row
        print("pred_combine.t():",pred)
        #print("size:",pred[0][0].type()) #5,12


        correct = pred.eq(target.view(1, -1).expand_as(pred)) #expend target to pred
        #print("correct:",correct)

        res = []
        for k in topk: #loop twice 1&5 
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)

            res.append(correct_k.mul_(100.0 / batch_size))
        return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

if __name__ == '__main__':
    main()