import xlrd
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import ConcatDataset
import torch
import random
import pandas as pd

class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class SeqDataset(Dataset):

    def __init__(self, path='../../deepinfomax/data/ecoli_expr_wy.xlsx', isTrain=True, isGpu=True, split_r=0.889,small=False):
        self.path = path
        exprs = []
        seqs = []
        with open(self.path) as f:
            for i, line in enumerate(f):
                if i == 0:
                    continue
                if small:
                    if i > 300:
                        break
                # active = list(map(lambda x: float(x), line.split(',')[0].split('\t')[1:]))
                # seqB = line.split(',')[1].split('\n')[0]
                active = list(map(lambda x: float(x), line.split(',')[1:31]))
                seqB = line.split(',')[31]
                print(len(seqB))
                if not len(seqB) == 2000:
                    continue
                exprs.append(active)
                seqs.append(seqB)
        random.seed(0)
        index = list(np.arange(len(seqs)))
        random.shuffle(index)
        self.pSeq = []
        self.expr = []
        self.isTrain = isTrain
        self.split_r = split_r
        self.isGpu = isGpu
        maxE = 1
        minE = 0
        if self.isTrain:
            start, end = 0, int(len(index)*self.split_r)
        else:
            start, end = int(len(index)*self.split_r), len(index)
        for i in range(start, end):
            self.pSeq.append(self.oneHot(seqs[i]))
            t = []
            for j in exprs[i]:
                t.append((j - minE)/(maxE - minE))
            self.expr.append(t)

    def oneHot(self, sequence):
        oh_dict = {'N':0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
        oh = np.zeros([5, len(sequence)])
        for i in range(len(sequence)):
            oh[oh_dict[sequence[i]], i] = 1
        return oh

    def __getitem__(self, item):
        X = self.pSeq[item][:, :]
        Z = self.expr[item]
        X = transforms.ToTensor()(X)
        X = torch.squeeze(X)
        X = X.float()
        Z = transforms.ToTensor()(np.asarray([Z]))
        Z = torch.squeeze(Z)
        Z = Z.float()
        if self.isGpu:
            X, Z = X.cuda(), Z.cuda()
        return {'x': X, 'z':Z}

    def __len__(self):
        return len(self.expr)


