import torch
import numpy as np
from utils import AverageMeter


class DataPrefetcher():
    def __init__(self, loader):
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch = next(self.loader)
        except StopIteration:
            self.batch = None
            return
        with torch.cuda.stream(self.stream):
            for k in self.batch:
                if k != 'meta':
                    self.batch[k] = self.batch[k].to(device=self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch = self.batch
        self.preload()
        return batch


def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()
    loss_all = 0
    for batch_idx, (data, labels) in enumerate(trainloader):
        if options['use_gpu']:
            if 'TI' in options['dataset']:
                data1 = data['img'].cuda(non_blocking=True)
                data2 = data['txt'].cuda(non_blocking=True)
                labels = labels.cuda()
            else:
                data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            x, y, loss_g = net(data1, data2, return_feature=True)
            _, loss_cls = criterion(x, y, labels)
            loss = loss_cls + loss_g
            print('loss:', loss.item())
            loss.backward()
            optimizer.step()

        losses.update(loss.item(), labels.size(0))
        if batch_idx+1 % options['print_freq'] == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f})".format(batch_idx+1, len(trainloader), losses.val, losses.avg))
        loss_all += losses.avg
    return loss_all