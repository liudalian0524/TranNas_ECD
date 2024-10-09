import keras
import pandas as pd
import autokeras as ak
import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
from keras.models import load_model
import sys
import os
from keras.layers import Dense

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


log_path = './Logs/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
# 日志文件名按照程序运行时间设置
log_file_name = "Zea_CNNDense.log"
# 记录正常的 print 信息
sys.stdout = Logger(log_file_name)
# 记录 traceback 异常信息
sys.stderr = Logger(log_file_name)


source_data = pd.read_csv("../../Data_P/AllData.csv")
source_pssm = source_data.iloc[:, 1:-1]
source_label = source_data.iloc[:, -1]
target_data = pd.read_csv("../../Data_P/Zea_Train.csv")
target_pssm = target_data.iloc[:, 1:-1]
target_label = target_data.iloc[:, -1]


# x = target_pssm.shape
# y = target_label.shape
# target_data = torch.cat([torch.tensor(target_pssm), target_label.reshape(target_label.shape[0], 1)], dim=1)

# df = pd.DataFrame(np.array(target_data))
# df.to_csv("targetDataCKPA2.csv")

source_train_pssm, source_test_pssm, source_train_label, source_test_label = train_test_split(source_pssm, source_label,
                                                                                              test_size=0.3,
                                                                                              random_state=300)
target_train_pssm, target_test_pssm, target_train_label, target_test_label = train_test_split(target_pssm, target_label,
                                                                                              test_size=0.7,
                                                                                              random_state=300)

source_train_set = (tf.data.Dataset.from_tensor_slices(
    (np.expand_dims(source_train_pssm, axis=1), np.asarray(source_train_label).astype('float32').reshape((-1, 1)))))
source_train_valid = tf.data.Dataset.from_tensor_slices((source_test_pssm, source_test_label))
target_train_set = tf.data.Dataset.from_tensor_slices(
    (np.expand_dims(target_train_pssm, axis=1), np.asarray(target_train_label).astype('float32').reshape((-1, 1))))
target_test_set = tf.data.Dataset.from_tensor_slices(
    (np.expand_dims(target_test_pssm, axis=1), np.asarray(target_test_label).astype('float32').reshape((-1, 1))))

# clf = ak.StructuredDataClassifier(max_trials=100)
# clf.fit(source_train_set, epochs=100)
#
# # predictions = clf.predict(source_train_valid)
# accuracy = clf.evaluate(target_test_set)
# print(f"Valid Accuracy: {accuracy}")
input_node = ak.Input()
cnn_block = ak.ConvBlock()(input_node)
dense_block=ak.DenseBlock()(cnn_block)
output_node = ak.ClassificationHead()(dense_block)
# for i in range(1,30,1):

# 建立模型
clf = ak.AutoModel(
    inputs=input_node,
    outputs=output_node,
    max_trials=30)

# 训练模型
clf.fit(source_train_set, epochs=100, batch_size=16)

# 评估模型
accuracy = clf.evaluate(target_test_set)

print(f'Target Test Acc:{accuracy}')
