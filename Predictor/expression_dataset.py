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

    def __init__(self, path='../../deepinfomax/data/ecoli_expr_wy.xlsx', isTrain=True, isGpu=True, split_r=0.889, seqL=200):
        self.path = path
        files = pd.read_csv(self.path)
        seqs = list(pd.read_csv(self.path)['realB'].head(-seqL))
        seqs = [sublist[-seqL:] if len(sublist) >= seqL else sublist for sublist in seqs]

        list_name = ['CNS E11.5', 'CNS E14', 'CNS E18', 'Large Intestine adult', 'adrenal adult', 'bladder adult', 'cerebellum adult', 'colon adult', 'cortex adult', 'duodenum adult', 'frontal Lobe adult', 'genital fat pad adult', 'heart adult', 'kidney adult', 'limb E14.5', 'liver E14', 'liver E14.5', 'liver E18', 'liver adult', 'lung adult', 'mammary gland adult', 'ovary adult', 'placenta adult', 'small intestine adult', 'spleen adult', 'stomach adult', 'subcutaneous fat pad adult', 'testis adult', 'thymus adult', 'whole brain E14.5']
        #list_name = ['CNS E14']
        lists = [files[ln] for ln in list_name]
        exprs = [list(row) for row in zip(*lists)] 
        #exprs = [sum(row)/30 for row in zip(*lists)]
        #exprs = list(files['expr'])
        
        seqsA = list(pd.read_csv(self.path)['realA'].head(-seqL))
        seqsA = [sublist[-seqL:] if len(sublist) >= seqL else sublist for sublist in seqsA]
        seqsB = list(pd.read_csv(self.path)['realB'].head(-seqL))
        seqsB = [sublist[-seqL:] if len(sublist) >= seqL else sublist for sublist in seqsB]
        indeces = [9880, 8741, 14647, 10327, 13160, 10043, 12800, 5648, 15742, 18225, 12139, 12669, 11003, 11843, 6150, 9286, 3194, 16881]
        with open('/mnt/wangbolin/code/DeepSEED/deepseed4mouse/data/input_promoters.txt', 'w', encoding='utf-8') as f:
            for i in range(18):
                print(i)
                f.write('>' + str(i+1) + '\n')
                f.write(seqsA[indeces[i]] + '\n')
                f.write(seqsB[indeces[i]] + '\n')
        aaaaaaaaaa
        random.seed(0)
        index = list(np.arange(len(seqs)))
        random.shuffle(index)
        self.pSeq = []
        self.expr = []
        self.isTrain = isTrain
        self.split_r = split_r
        self.isGpu = isGpu
        if self.isTrain:
            start, end = 0, int(len(index)*self.split_r)
        else:
            start, end = int(len(index)*self.split_r), len(index)
        for i in range(start, end):
            self.pSeq.append(self.oneHot(seqs[i]))
            self.expr.append(exprs[i])

    def oneHot(self, sequence):
        oh_dict = {'A': 0, 'T': 1, 'C': 2, 'G': 3, 'N':4}
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
        Z = transforms.ToTensor()(np.asarray([[Z]]))
        Z = torch.squeeze(Z)
        Z = Z.float()
        if self.isGpu:
            X, Z = X.cuda(), Z.cuda()
        return {'x': X, 'z':Z}

    def __len__(self):
        return len(self.expr)


