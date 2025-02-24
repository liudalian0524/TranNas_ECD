import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist, cosine

# 读取CSV文件，第一列为行索引，第一行不是列索引
file_path = '../data3/Physcomitrella_Pack.csv'
data = pd.read_csv(file_path)

# 假设最后一列是标签，标签为1是阳性样品，标签为0是未标记样品
# labels = data.iloc[:, -1]
# features = data.iloc[:, 1:-1]
features = data.iloc[:, 1:-1]
labels = data.iloc[:, -1]
#保留原始索引
original_index = features.index
# 标准化特征矩阵
# scaler = StandardScaler()
# features_scaled = scaler.fit_transform(features)

# 分割数据为阳性样品和未标记样品
positive_samples = features[labels == 1]
negative_samples = features[labels == 0]
# 保留未标记样品的原始索引
negative_samples_index = original_index[labels == 0]

# 对负样品进行K-means聚类，并获取质心向量D
kmeans_negative = KMeans(n_clusters=1, random_state=42).fit(negative_samples)
D = kmeans_negative.cluster_centers_

# 对阳性样品进行K-means聚类，并获取质心向量C
kmeans_positive = KMeans(n_clusters=1, random_state=42).fit(positive_samples)
C = kmeans_positive.cluster_centers_

def cosine_similarity(v1, v2):
    return 1 - cosine(v1, v2)

def euclidean_distance(v1, v2):
    return np.linalg.norm(v1 - v2)

# 计算每个负样本到正类和负类质心的余弦相似度
cosine_sim_to_C = np.apply_along_axis(lambda x: cosine_similarity(x, C.flatten()), 1, negative_samples)
cosine_sim_to_D = np.apply_along_axis(lambda x: cosine_similarity(x, D.flatten()), 1, negative_samples)

# 根据相似度将负样本分类为近似正类P和近似负类N
P = negative_samples[cosine_sim_to_C > cosine_sim_to_D]
N = negative_samples[cosine_sim_to_C <= cosine_sim_to_D]

# 计算近似正类P和近似负类N的质心向量E和F
kmeans_P = KMeans(n_clusters=1, random_state=42).fit(P)
E = kmeans_P.cluster_centers_

kmeans_N = KMeans(n_clusters=1, random_state=42).fit(N)
F = kmeans_N.cluster_centers_
euclidean_dist_to_E = np.apply_along_axis(lambda x: euclidean_distance(x, E.flatten()), 1, negative_samples)
euclidean_dist_to_F = np.apply_along_axis(lambda x: euclidean_distance(x, F.flatten()), 1, negative_samples)
P_prime = negative_samples[euclidean_dist_to_E < euclidean_dist_to_F]
N_prime = negative_samples[euclidean_dist_to_E >= euclidean_dist_to_F]
G = KMeans(n_clusters=1, random_state=42).fit(P_prime).cluster_centers_
H = KMeans(n_clusters=1, random_state=42).fit(N_prime).cluster_centers_
# 迭代更新
tolerance = 1e-3
while np.linalg.norm(E - G) <= tolerance and np.linalg.norm(F - H) <= tolerance:
    E, F = G, H
    euclidean_dist_to_E = np.apply_along_axis(lambda x: euclidean_distance(x, E.flatten()), 1, negative_samples)
    euclidean_dist_to_F = np.apply_along_axis(lambda x: euclidean_distance(x, F.flatten()), 1, negative_samples)

    P_prime = negative_samples[euclidean_dist_to_E < euclidean_dist_to_F]
    N_prime = negative_samples[euclidean_dist_to_E >= euclidean_dist_to_F]

    G = KMeans(n_clusters=1, random_state=42).fit(P_prime).cluster_centers_
    H = KMeans(n_clusters=1, random_state=42).fit(N_prime).cluster_centers_
N = N_prime
N_prime_index = negative_samples_index[euclidean_dist_to_E >= euclidean_dist_to_F]
print(N_prime.shape)
print(N_prime_index)
selected_data = data.loc[N_prime_index]
print(selected_data)
selected_data.to_csv('../data3/Physcomitrella_Pack_N.csv')
# # 打印最终的质心向量
# print("Final Centroid for positive samples (E):")
# print(E)
# print("\nFinal Centroid for negative samples (F):")
# print(F)