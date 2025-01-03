from torch.utils.data import ConcatDataset, DataLoader
import numpy as np
import torchvision.transforms as transforms
from torch import squeeze
import pandas as pd


class Dataset(object):

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __add__(self, other):
        return ConcatDataset([self, other])


class LoadData(Dataset):

    def __init__(self, path='data/ecoli_100_space_fix.csv', split_r=0.9, is_train=True, gpu_ids='0',small=False):
        actives = []
        realB = []
        realA = []
        with open(path) as f:
            for i, line in enumerate(f):
                if i==0:
                    continue
                if small:
                    if i>300:
                        break
                active = map(lambda x:float(x), line.split(',')[0].split('\t')[1:])
                seqB = line.split(',')[1]
                if not len(seqB)==2001:
                    continue
                seqA = 'M'*2000
                actives.append(active)
                realB.append(seqB)
                realA.append(seqA)
        data_size = len(realB)
        split_idx = int(data_size * split_r)
        noise_dim = 128
        self.gpu_ids = gpu_ids
        if is_train:
            st, ed = 0, split_idx
        else:
            st, ed = split_idx, data_size
        self.storage, self.input_seq = [], []
        for i in range(st, ed, 1):
            self.storage.append(one_hot(realB[i].split('\n')[0].upper()))
            self.input_seq.append(backbone_one_hot(realA[i].split('\n')[0].upper()))

    def __getitem__(self, item):
        in_seq, label_seq = transforms.ToTensor()(self.input_seq[item]), transforms.ToTensor()(self.storage[item])
        if len(self.gpu_ids) > 0:
            return {'in': in_seq[0, :].float().cuda(), 'out': squeeze(label_seq).float().cuda()}
        else:
            return {'in': in_seq[0, :].float(), 'out': squeeze(label_seq).float()}

    def __len__(self):
        return len(self.storage)


def one_hot(seq):
    charmap = {'N':0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        encoded[charmap[seq[i]], i] = 1
    return encoded


def backbone_one_hot(seq):
    charmap = {'N': 0, 'A': 1, 'T': 2, 'C': 3, 'G': 4}
    encoded = np.zeros([len(charmap), len(seq)])
    for i in range(len(seq)):
        if seq[i] == 'M':
            ###改正态分布
            encoded[:, i] = np.random.randn(5)
        else:
            encoded[charmap[seq[i]], i] = 1
    return encoded