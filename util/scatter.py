import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

source_data = pd.read_csv("../newFeaturePack/ArabFeature.csv")
source_pssm = source_data.iloc[:, 1:-1]
source_label = source_data.iloc[:, -1]
source_pssm=np.array(source_pssm)
source_label=np.array(source_label)
target_data = pd.read_csv("../newFeaturePack/OryzaFeature.csv")
target_pssm = target_data.iloc[:, 1:-1]
target_label = target_data.iloc[:, -1]
target_pssm=np.array(target_pssm)
target_label=np.array(target_label)
target_data2 = pd.read_csv("../newFeaturePack/SlycFeature.csv")
target_pssm2 = target_data2.iloc[:, 1:-1]
target_label2 = target_data2.iloc[:, -1]
target_pssm2=np.array(target_pssm2)
target_label2=np.array(target_label2)

from sklearn.decomposition import PCA
# from sklearn.preprocessing import MinMaxScaler
#
# source_pssm = MinMaxScaler().fit_transform(source_pssm)
# target_pssm = MinMaxScaler().fit_transform(target_pssm)
# target_pssm2 = MinMaxScaler().fit_transform(target_pssm2)
pca = PCA(n_components=2)  # 选择合适的主成分
pca2 = PCA(n_components=2)  # 选择合适的主成分
pca3 = PCA(n_components=2)  # 选择合适的主成分
#
# # 数目
source_pssm_pca = pca.fit_transform(source_pssm)
target_pssm_pca1 = pca.fit_transform(target_pssm)
target_pssm_pca2 = pca.fit_transform(target_pssm2)
data = pd.DataFrame(source_pssm_pca)
data2 = pd.DataFrame(target_pssm_pca1)
data3 = pd.DataFrame(target_pssm_pca2)

# 创建一个 DataFrame，每一列代表一个特征
# data = pd.DataFrame(np.random.rand(num_samples, num_features), columns=[f"Feature_{i+1}" for i in range(num_features)])

# 添加一个用于颜色的随机列
data['Label'] = 'Arabidopsis'
data2['Label'] = 'Oryza'
data3['Label'] = 'Solanum'
data4 = np.concatenate((data,data2,data3),axis=0)
data_label= np.concatenate((data['Label'],data2['Label'],data3['Label']),axis=0)


# 选择前两个特征用于散点图
# selected_features1 = np.random.choice(len(data4),size=2,replace=False)
plt.figure(figsize=(10,6))
species=['Arabidopsis','Oryza','Solanum']
for specie in species:
    species_data = data4[data_label==specie]
    plt.scatter(species_data[:,0],species_data[:,1],label=f'{specie}')

plt.legend()
plt.show()
