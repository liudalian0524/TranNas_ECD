import pandas as pd
import autokeras as ak
import numpy as np
# import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os
import util.evaulate as evaulate

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# print(torch.cuda.device_count())
# print("----------")
# if torch.cuda.is_available():
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'

# 设置TensorFlow使用GPU
# gpus = tf.config.experimental.list_physical_devices('GPU')
# print(gpus)
# if gpus:
#     try:
#         # 设置GPU内存增长
#         for gpu in gpus:
#             tf.config.experimental.set_memory_growth(gpu, True)
#     except RuntimeError as e:
#         print(e)


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
log_file_name = "OryzaToArab.log"
# 记录正常的 print 信息
sys.stdout = Logger(log_file_name)
# 记录 traceback 异常信息
sys.stderr = Logger(log_file_name)

source_data = pd.read_csv("../../../Data_P/Oryza_Pack.csv")
source_pssm = source_data.iloc[:, 1:-1]
source_label = source_data.iloc[:, -1]
target_data = pd.read_csv("../../../Data_P/Arab_Pack.csv")
target_pssm = target_data.iloc[:, 1:-1]
target_label = target_data.iloc[:, -1]

# source_result = StandardScaler().fit_transform(source_pssm)
# target_result = StandardScaler().fit_transform(target_pssm)

source_train, source_val, source_train_label, source_val_label = train_test_split(source_pssm, source_label,
                                                                                  test_size=0.3, random_state=300)
target_train, target_val, target_train_label, target_val_label = train_test_split(target_pssm, target_label,
                                                                                  test_size=0.7, random_state=300)

train_set = tf.data.Dataset.from_tensor_slices((source_train, source_train_label))
test_set = tf.data.Dataset.from_tensor_slices((target_val, target_val_label))


clf = ak.StructuredDataClassifier(max_trials=30)
clf.fit(train_set, epochs=100)

model = clf.export_model()
model.save("modelOA.h5")

predictions = clf.predict(test_set)
# # 计算准确率
accuracy = clf.evaluate(test_set)
pre, recall, f1, mcc, acc = evaulate.myEvaulate2(target_val_label, predictions)
print(f"acc={accuracy},acc={acc},pre={pre},recall={recall},f1={f1},mcc={mcc}")
