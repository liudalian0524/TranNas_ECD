import csv

def csv_to_fasta(csv_file, fasta_file):
    with open(csv_file, 'r') as csvfile, open(fasta_file, 'w') as fastafile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header
        for row in reader:
            header, sequence, label = row
            fastafile.write(f'>{header}\n')
            fastafile.write(f'{sequence}\n')

# 使用示例
csv_file = '../../newData/Zea_Heat.csv'
fasta_file = '../../newData/Zea_Heat.fasta'
csv_to_fasta(csv_file, fasta_file)