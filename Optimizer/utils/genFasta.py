import pandas as pd


def csv2fasta(csv_path, data_path, data_name):
    path = csv_path
    results = pd.read_csv(path)
    fakeB = list(results['fakeB'])
    realB = list(results['realB'])
    f2 = open(data_path + data_name + '_realB.fasta','w')
    j = 0
    for i in realB:
        f2.write('>sequence_generate_'+str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()
    f2 = open(data_path + data_name + '_fakeB.fasta','w')
    j = 0
    for i in fakeB:
        f2.write('>sequence_generate_'+str(j) + '\n')
        f2.write(i + '\n')
        j = j + 1
    f2.close()


def main():
    file_name = '../results/lac100_2021-03-05-13-00-03_results.csv'
    csv2fasta(file_name, '../results/', 'before')

if __name__ == '__main__':
    main()