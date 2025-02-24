import pandas as pd


df = pd.read_csv('../../data/Zea_Heat.csv')  # 替换为你的文件名
df2 = pd.read_csv('../../data/Zea_Neg.csv')
# 读取特定列
name = df['header'].tolist()
sequence = df['sequence'].tolist()

name2 = df2['header'].tolist()
sequence2 = df2['sequence'].tolist()

name3 = []
sequence3 = []
label3 = []

i = 0
for seq in sequence:
    if seq in sequence3:
        print(name[i])
        i = i + 1
        continue
    elif 'X' in seq:
        i = i + 1
        continue
    else:
        name3.append(name[i])
        sequence3.append(seq)
        label3.append(1)
        i = i + 1

length =len(sequence3) * 3
i = 0
for seq in sequence2:
    if seq in sequence3:
        print(name2[i])
        i = i + 1
        continue
    elif 'X' in seq:
        i = i + 1
        continue
    else:
        name3.append(name2[i])
        sequence3.append(seq)
        label3.append(0)
        i = i + 1

    if len(sequence3)>=length:
        break

df = pd.DataFrame(list(zip(name3, sequence3,label3)), columns=['name', 'sequence','label'])

# 将 DataFrame 写入 CSV 文件
df.to_csv('../../data/Zea_Heat_All.csv', index=False)