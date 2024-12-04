import os
import csv
import time
import torch
import random
import argparse
import datetime
import importlib
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from torch.optim import lr_scheduler

from core import *
from utils import *
from train import *

import warnings
warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser("Training")
parser.add_argument('--seed', type=int, default=71902)
parser.add_argument('--gpu', type=str, default='')
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--outf', type=str, default='./log/')
parser.add_argument('--eval-path', type=str, help="path", default='')
parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--loss', type=str, default='mrn_loss')

# Dataset
parser.add_argument('--dataroot', type=str, default='')
parser.add_argument('--dataset', type=str, default='')

# optimization
parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--stepsize', type=int, default=60)
parser.add_argument('--max-epoch', type=int, default=80)
parser.add_argument('--eval-freq', type=int, default=10)
parser.add_argument('--print-freq', type=int, default=10)
parser.add_argument('--batch-size', type=int, default=128)

# model
parser.add_argument('--num_exp', type=int, default=15)
parser.add_argument('--model', type=str, default='MRN')


def main(options):
    if options['seed'] is not None:
        random.seed(args.seed)
        torch.manual_seed(options['seed'])  
        torch.cuda.manual_seed(options['seed']) 
        torch.backends.cudnn.deterministic = True
        
    torch.cuda.set_device(int(options['gpu']))
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(options['seed'])

    # Dataset
    print("{} Preparation".format(options['dataset']))
    data = getLoader(options)
    trainloader, testloader, outloader = data.train_loader, data.test_loader, data.out_loader
    options['num_classes'] = data.num_classes

    print("Creating model: {}".format(options['model']))
    if 'flo' or 'cub' in options['dataset']:
        net = MRN(options = options, num_classes = options['num_classes'], text_dim = 1024)
    elif 'food' in options['dataset']:
        net = MRN(options = options, num_classes = options['num_classes'], text_dim = 300)

    options.update(
        {
            'feat_dim': options['feat_dim'],
            'use_gpu' : use_gpu
        }
    )

    Loss = importlib.import_module('loss.'+options['loss'])
    criterion = getattr(Loss, options['loss'])(**options)

    if use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()

    model_path = os.path.join(options['outf'], 'models', options['dataset'])
    temp = '_{}_{}_{}'.format(options['model'], options['loss'], options['lr'])
    model_path += temp

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    file_name = '{}_{}_{}'.format(options['model'], options['loss'], options['item'])
    
    params_list = [{'params': net.parameters()}, {'params': criterion.parameters()}]
    optimizer = torch.optim.Adam(params_list, lr=options['lr'])
    scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80])

    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']))
        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)
        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'])
            results = test(net, criterion, testloader, outloader, epoch=epoch, **options)
            print("Acc (%): {:.3f}\t AUROC (%): {:.3f}\t OSCR (%): {:.3f}\t".format(results['ACC'], results['AUROC'], results['OSCR']))

        save_networks(net, model_path, file_name, criterion=criterion)
        if options['stepsize'] > 0:
            scheduler.step()

    return results

if __name__ == '__main__':
    args    = parser.parse_args()
    options = vars(args)
    results = dict()
    
    for i in range(len(splits[options['dataset']])):
        known = splits[options['dataset']][len(splits[options['dataset']])-i-1]
        if 'flo' in options['dataset']:
            unknown = list(set(list(range(0, 102))) - set(known))
        elif 'food' in options['dataset']:
            unknown = list(set(list(range(0, 101))) - set(known))
        options.update(
            {
                'item':     i,
                'known':    known,
                'unknown':  unknown,
            }
        )

        dir_path = os.path.join(options['outf'], 'results')
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        file_name = '{}_{}_{}_{}'.format(options['dataset'], options['model'], options['loss'], options['lr'])
        file_name = file_name + '.csv'

        res = main(options)
        res['unknown']  = unknown
        res['known']    = known
        results[str(i)] = res
        df = pd.DataFrame(results)
        df.to_csv(os.path.join(dir_path, file_name))