import os
from Bio import SeqIO
chromosomes = [str(i) for i in range(1,18)]
data_route = "/Data/Chlamydomonas_reinhardtii"
gene_result = os.path.join(data_route,"gene_result.xls")
seq_file = os.path.join(data_route,"sequence.fasta")
test = []
ero = []
seq_dict = {}
for rec in SeqIO.parse(seq_file,"fasta"):
    home = int(rec.id[6:8])-61
    seq_dict[home]=str(rec.seq)
out_list = []
with open(gene_result) as gene_f:
    for i,line in enumerate(gene_f):
        if i<=0:
            continue
        feat = line.strip().split('\t')
        ###17470有效，409不属于17染色体的，其中273个长度不足16的。
        if len(feat)<11:
            continue
        if feat[10] not in chromosomes:
            continue

        chromosome = int(feat[10])
        gene_id = feat[2]
        orientation = feat[14]
        start = int(feat[12])
        end = int(feat[13])
        ###从0开始，要减一
        pro_gene_seq = seq_dict[chromosome][start-1:end]
        if orientation=='plus':
            promoter = seq_dict[chromosome][start-2001:start-1]
        else:
            promoter = seq_dict[chromosome][end:end+2000]
        if not len(promoter)==2000:
            print(len(seq_dict[chromosome]))
            print(promoter)
        out_list.append([gene_id,str(chromosome),orientation,promoter])
print(len(out_list))
with open(os.path.join(data_route,"promoters.csv"), 'w') as out_f:
    out_f.write("gene_id,chromosome,orientation,promoter\n")
    for i,rec in enumerate(out_list):
        out_f.write(','.join(rec))
        if not i==len(out_list)-1:
            out_f.write('\n')
#         if not len(feat)==16:
#             ero.append(feat)
#             print(feat)
#             continue
#         test.append(feat)
# print(len(ero))