import pandas as pd

def fasta_to_csv(input_file, output_file):
    sequences = {'header': [], 'sequence': []}
    current_sequence = ''
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                if current_sequence:
                    sequences['sequence'].append(current_sequence)
                    current_sequence = ''
                sequences['header'].append(line[1:])
            else:
                current_sequence += line
        if current_sequence:
            sequences['sequence'].append(current_sequence)

    df = pd.DataFrame(sequences)
    df.to_csv(output_file, index=False)

# 用法示例
fasta_to_csv('../../HeatData/Zea_Neg_Result.fasta', '../../data/Zea_Neg.csv')