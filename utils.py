import os
import sys
import torch
import errno
import numpy as np
import os.path as osp

from dataprocess.osr_loader import *

def mkdir_if_missing(directory):
    if not osp.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise


class AverageMeter(object):
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
        

class Logger(object):
    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(os.path.dirname(fpath))
            self.file = open(fpath, 'w')

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


def get_txt_name(m_file):
    f = open(m_file, 'r')
    content = f.read()
    f.close()
    return content.strip().split('\n')


def allTXT(npynamelist,dataP):
    text_names = npynamelist
    allTxtList = []
    num = 0
    for idex,item in enumerate(text_names):
        nowclass = np.load(os.path.join(dataP, item))
        allTxtList.append(nowclass)
        num = num + len(nowclass)
    return allTxtList


def getDicToTXT_cub():
    pathdic = {}
    with open('dataprocess/cub/img2text_cub.txt', 'r') as f:
        for line in f:
            a = list(line.strip('\n').split())
            a1 = a[0]
            pathdic[a1] = [int(a[1]), int(a[2])]
    return pathdic


def getDicToTXT_Flo():
    pathdic = {}
    with open('dataprocess/flower/img2text_flower.txt', 'r') as f:
        for line in f:
            a = list(line.strip('\n').split())
            a1 = a[0]
            pathdic[a1] = [int(a[1]), int(a[2])]
    return pathdic


def save_networks(networks, result_dir, name='', loss='', criterion=None):
    mkdir_if_missing(osp.join(result_dir, 'checkpoints'))
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    torch.save(weights, filename)
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        torch.save(weights, filename)


def load_networks(networks, result_dir, name='', loss='', criterion=None):
    weights = networks.state_dict()
    filename = '{}/checkpoints/{}_{}.pth'.format(result_dir, name, loss)
    checkpoint = torch.load(filename)
    new_state_dict = {}
    if(criterion == None):
        for k, v in checkpoint.items():
            new_state_dict[k[7:]] = v
    else:
        for k, v in checkpoint.items():
            new_state_dict[k] = v
    networks.load_state_dict(new_state_dict)
    if criterion:
        weights = criterion.state_dict()
        filename = '{}/checkpoints/{}_{}_criterion.pth'.format(result_dir, name, loss)
        criterion.load_state_dict(torch.load(filename))
        
    return networks, criterion


def getLoader(options):
    if ('cub_img' in options['dataset']):
        Data = CubImg_OSR(known=options['known'], unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif ('cub_txt' in options['dataset']):
        Data = Cubtxt_OSR(known=options['known'], unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif('cubTI' in options['dataset']):
        Data = CubimageImageTxt_OSR(known=options['known'], unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif ('flo_img' in options['dataset']):
        Data = Floimage_OSR(known=options['known'], unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif ('flo_txt' in options['dataset']):
        Data = Flotxt_OSR(known=options['known'], unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif ('floTI' in options['dataset']):
        Data = FloImageTxt_OSR(known=options['known'], unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif ('food_img' in options['dataset']):
        Data = Food101image_OSR(known=options['known'], unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif ('food_txt' in options['dataset']):
        Data = Food101txt_OSR(known=options['known'], unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'])
        trainloader, testloader, outloader = Data.train_loader, Data.test_loader, Data.out_loader
    elif ('foodTI' in options['dataset']):
        Data = Food101ImageTxt_OSR(known=options['known'], unknown=options['unknown'], dataroot=options['dataroot'], batch_size=options['batch_size'])
    
    return Data