import torch
import torch.nn as nn
from torch.nn import init
from pro_data import LoadData
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.autograd as autograd
from tensor2seq import save_sequence, tensor2seq, reserve_percentage
from transfer_fasta import csv2fasta
import utils
import matplotlib
matplotlib.use('Agg')
import argparse
import os
from tqdm import tqdm
import torch.optim as optim

def get_infinite_batches(data_loader):
    while True:
        for i, data in enumerate(data_loader):
            yield data

class MyConv(nn.Module):
    def __init__(self, in_channels, out_channels, padding=(2, 1)):
        super(MyConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(5, 3), stride=1, padding=padding)

    def forward(self, x):
        out = self.conv(x)
        return out

class TransformerModel(nn.Module):
    def __init__(self, input_dim=2000, d_model=512, nhead=8, num_coder_layers=1):
        super(TransformerModel, self).__init__()
        self.cov1 = MyConv(in_channels=5, out_channels=5)
        self.embedding = nn.Linear(input_dim, d_model)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.encoder_layer, num_coder_layers)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_coder_layers)
        self.fc = nn.Linear(d_model, input_dim)
        self.input_dim = input_dim
        self.d_model = d_model

    def forward(self, src, tgt):
        # 对src进行卷积等预处理
        src_out = src.unsqueeze(-1)
        src_out = self.cov1(src_out)
        src_out = src_out.squeeze(-1)
        src = src + src_out

        src = self.embedding(src)
        src = src.permute(1, 0, 2)
        memory = self.encoder(src)
        output1 = memory.permute(1, 0, 2)
        output1 = self.fc(output1)

        # 对tgt进行处理
        tgt_out = tgt.unsqueeze(-1)
        tgt_out = self.cov1(tgt_out)
        tgt_out = tgt_out.squeeze(-1)
        tgt = tgt + tgt_out
        tgt = self.embedding(tgt)
        tgt = tgt.permute(1, 0, 2)

        output = self.decoder(tgt, memory)
        output = output.permute(1, 0, 2)

        output2 = self.fc(output)
        return output1, output2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, default='random_seq_v2', help="path to data")
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument("--seqL", default=2000, type=int)
    parser.add_argument("--gpuid", default='0')
    parser.add_argument("--n_critics", default=5, type=int)
    parser.add_argument("--n_iters", default=10000, type=int)
    parser.add_argument("--nhead", default=8, type=int)
    parser.add_argument("--num_layers", default=1, type=int)
    parser.add_argument("--d_model", default=256, type=int)
    parser.add_argument("--weight", default=0.3, type=float)
    args = parser.parse_args()
    
    data_name = args.data_name 
    batch_size = args.batch_size
    seqL = args.seqL
    n_critics = args.n_critics
    n_iters = args.n_iters
    nhead = args.nhead
    num_layers = args.num_layers
    d_model = args.d_model
    weight = args.weight
    

    folder_name =  '/mnt/wangbolin/code/data/Generator_results/' + 'coFcTrans' + '_nlayers' + str(num_layers) + '_w'  + str(weight) + '_dmodel' + str(d_model) + \
         '_nhead'+ str(nhead) +  'bs' + str(batch_size) + '_seqL' + str(seqL) + '_ncritics' + str(n_critics) + '_niters' + str(n_iters) + '_' + data_name + '/'
    try:
        os.mkdir(folder_name)
        print(f"文件夹 {folder_name} 创建成功！")
    except FileExistsError:
        print(f"文件夹 {folder_name} 已存在，无需重复创建。")

    ckpt_path = folder_name + 'ckpt/'
    try:
        os.mkdir(ckpt_path)
        print(f"文件夹 {ckpt_path} 创建成功！")
    except FileExistsError:
        print(f"文件夹 {ckpt_path} 已存在，无需重复创建。")
        
    cache_path = folder_name + 'cache/'
    try:
        os.mkdir(cache_path)
        print(f"文件夹 {cache_path} 创建成功！")
    except FileExistsError:
        print(f"文件夹 {cache_path} 已存在，无需重复创建。")
        
    cache_figure_path = cache_path + 'figure/'
    try:
        os.mkdir(cache_figure_path)
        print(f"文件夹 {cache_figure_path} 创建成功！")
    except FileExistsError:
        print(f"文件夹 {cache_figure_path} 已存在，无需重复创建。")
    split_r = 0.8
    train_data, test_data = DataLoader(LoadData(is_train=True, path='/mnt/wangbolin/code/data/data/{}.csv'.format(data_name), split_r=split_r, num=seqL),
                                       batch_size=batch_size, shuffle=True), DataLoader(LoadData(is_train=False, 
                                       path='/mnt/wangbolin/code/data/data/{}.csv'.format(data_name), split_r=split_r, num=seqL),
                                       batch_size=batch_size)
    train_data = get_infinite_batches(train_data)
    logger = utils.get_logger(log_path=folder_name, name='training')
    logger.info('num_layers: {}, nhead:{}, d_model:{}, weight: {}'.format(num_layers, nhead, d_model, weight))
    logger.info('data_name: {}, n_iters:{}, n_critics:{}, batch_size: {}, seqL:{}'.format(data_name, n_iters, n_critics, batch_size, seqL))
    
    device = torch.device('cuda:{}'.format(args.gpuid)) if args.gpuid else torch.device('cpu')
    torch.cuda.set_device(device)
    model = TransformerModel(input_dim=seqL, d_model=d_model, nhead=nhead, num_coder_layers=num_layers).to(device)
    loss1, loss2, loss = 0, 0, 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    for i in tqdm(range(n_iters)):
        #train discriminators
        for j in range(n_critics):
            _data = train_data.__next__()
            in_data = _data['in'].to(device)
            out_data= _data['out'].to(device)
            
            output1, output2 = model(in_data, out_data)
            loss1 = criterion(output1, out_data)
            loss2 = criterion(output2, out_data)
            loss = loss1 + weight * loss2
            print(i, j, loss1.item(), loss2.item())
            
            optimizer.zero_grad()  
            loss.backward()        
            optimizer.step()
        
        if i == 9 :
            best_loss1, best_loss2, best_loss = loss1, loss2, loss
            
        if i % 10 == 9:
            tensorSeq, tensorInput, tensorRealB = [], [], []
            for j, eval_data in enumerate(test_data):
                with torch.no_grad():
                    in_data = _data['in'].to(device)
                    out_data= _data['out'].to(device)
                    output1, output2 = model(in_data, out_data)
                    tensorSeq.append(output1)
                    tensorInput.append(eval_data['in'])
                    tensorRealB.append(eval_data['out'])
            logger.info('Training: iters: {}, loss1: {}, loss2:{}, loss:{}'.format(i, loss1 / 10 / n_critics, loss2 / 10/ n_critics, loss / 10/ n_critics))
            logger.info('Testing: reserve percentage: {}%'.format(reserve_percentage(tensorInput, tensorSeq)))
            csv_name = save_sequence(tensorSeq, tensorInput, tensorRealB, save_path=cache_path, name='inducible_', cut_r=0.1)
            A_dict_valid, A_dict_ref, T_dict_valid, T_dict_ref = utils.polyAT_freq(csv_name, '/mnt/wangbolin/code/data/data/{}.csv'.format(data_name))
            logger.info('polyA valid AAAAA:{} AAAAAA:{} AAAAAAA:{} AAAAAAAA:{}'.format(A_dict_valid['AAAAA'],
                                                                                 A_dict_valid['AAAAAA'],
                                                                                 A_dict_valid['AAAAAAA'],
                                                                                 A_dict_valid['AAAAAAAA']))
            logger.info('polyA ref AAAAA:{} AAAAAA:{} AAAAAAA:{} AAAAAAAA:{}'.format(A_dict_ref['AAAAA'],
                                                                                 A_dict_ref['AAAAAA'],
                                                                                 A_dict_ref['AAAAAAA'],
                                                                                 A_dict_ref['AAAAAAAA']))
            logger.info('polyT valid TTTTT:{} TTTTTT:{} TTTTTTT:{} TTTTTTTT:{}'.format(T_dict_valid['TTTTT'],
                                                                                 T_dict_valid['TTTTTT'],
                                                                                 T_dict_valid['TTTTTTT'],
                                                                                 T_dict_valid['TTTTTTTT']))
            logger.info('polyT ref TTTTT:{} TTTTTT:{} TTTTTTT:{} TTTTTTTT:{}'.format(T_dict_ref['TTTTT'],
                                                                               T_dict_ref['TTTTTT'],
                                                                               T_dict_ref['TTTTTTT'],
                                                                               T_dict_ref['TTTTTTTT']))
            #utils.kmer_frequency(csv_name, '../data/{}.csv'.format(data_name), k=4, save_path=cache_figure_path, save_name=data_name + '_' + str(i))
            csv2fasta(csv_name, cache_path + 'gen_', 'iter_{}'.format(i))
            if best_loss1 > loss1 :
                best_loss1 = loss1
                torch.save(model, folder_name + '/ckpt/' + 'best' + '.pth')
            else :
                torch.save(model, folder_name + '/ckpt/' + 'latest' + '.pth')
            loss1, loss2, loss = 0, 0, 0

if __name__ == '__main__':
    main()
