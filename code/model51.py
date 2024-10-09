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

# x = target_pssm.shape
# y = target_label.shape
# target_data = torch.cat([torch.tensor(target_pssm),target_label.reshape(target_label.shape[0],1)],dim=1)
#
# df = pd.DataFrame(np.array(target_data))
# df.to_csv("targetDataPAAC.csv")

source_train_pssm, source_test_pssm, source_train_label, source_test_label = train_test_split(source_pssm,
                                                                                              source_label,
                                                                                              test_size=0.3,
                                                                                              random_state=300)
target_train_pssm, target_test_pssm, target_train_label, target_test_label = train_test_split(target_pssm,
                                                                                              target_label,
                                                                                              test_size=0.7,
                                                                                              random_state=300)
#
source_train_set = tf.data.Dataset.from_tensor_slices(
    (source_train_pssm, np.asarray(source_train_label).astype('float32').reshape((-1, 1))))
source_train_valid = tf.data.Dataset.from_tensor_slices((source_test_pssm, source_test_label))
target_train_set = tf.data.Dataset.from_tensor_slices(
    (target_train_pssm, np.asarray(target_train_label).astype('float32').reshape((-1, 1))))
target_test_set = tf.data.Dataset.from_tensor_slices(
    (target_test_pssm, np.asarray(target_test_label).astype('float32').reshape((-1, 1))))

# clf = ak.StructuredDataClassifier(max_trials=100)
# clf.fit(source_train_set, epochs=100)
#
# # predictions = clf.predict(source_train_valid)
# accuracy = clf.evaluate(target_test_set)
# print(f"Valid Accuracy: {accuracy}")
# for i in range(1,20,1):
clf = ak.StructuredDataClassifier(max_trials=30)
clf.fit(source_train_set, epochs=100)

predictions = clf.predict(target_test_set)
accuracy = clf.evaluate(target_test_set)
print(f"Valid Accuracy: {accuracy}")
