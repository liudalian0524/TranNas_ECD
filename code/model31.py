import pandas as pd
import autokeras as ak
import numpy as np
import torch
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter
import util.evaulate as evaulate
import sys
import os

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
# 日志文件名按照程序运行时间设置model11.py
log_file_name = "Glycine.log"
# 记录正常的 print 信息
sys.stdout = Logger(log_file_name)
# 记录 traceback 异常信息
sys.stderr = Logger(log_file_name)

source_data = pd.read_csv("../../../Data_P/AllData.csv")
source_pssm = source_data.iloc[:, 1:-1]
source_label = source_data.iloc[:, -1]
target_data = pd.read_csv("../../../Data_P/Glycine_Train.csv")
target_pssm = target_data.iloc[:, 1:-1]
target_label = target_data.iloc[:, -1]
target_data2 = pd.read_csv("../../../Data_P/Glycine_Test.csv")
target_pssm2 = target_data2.iloc[:, 1:-1]
target_label2 = target_data2.iloc[:, -1]


source_train_pssm, source_test_pssm, source_train_label, source_test_label = train_test_split(source_pssm,source_label,test_size=0.3, random_state=300)
target_train_pssm, target_test_pssm, target_train_label, target_test_label = train_test_split(target_pssm,target_label,test_size=0.7, random_state=300)

source_train_pssm = pd.concat([source_train_pssm, target_train_pssm], axis=0)
source_train_label = pd.concat([source_train_label, target_train_label], axis=0)


source_train_set = tf.data.Dataset.from_tensor_slices((source_train_pssm,source_train_label))
source_train_valid = tf.data.Dataset.from_tensor_slices((source_test_pssm,source_test_label))
target_test_set = tf.data.Dataset.from_tensor_slices((target_test_pssm,target_test_label))
target_test_set2 = tf.data.Dataset.from_tensor_slices((target_pssm2,target_label2))

clf = ak.StructuredDataClassifier(max_trials=30)
clf.fit(source_train_set, epochs=100)

predictions = clf.predict(target_test_set)
predictions2 = clf.predict(target_test_set2)
# # 计算准确率
accuracy = clf.evaluate(target_test_set)
accuracy2 = clf.evaluate(target_test_set2)
pre, recall, f1, mcc, acc = evaulate.myEvaulate2(target_test_label, predictions)
pre2, recall2, f12, mcc2, acc2 = evaulate.myEvaulate2(target_label2, predictions2)
print(f"acc={accuracy},acc={acc},pre={pre},recall={recall},f1={f1},mcc={mcc}")
print(f"acc={accuracy2},acc={acc2},pre={pre2},recall={recall2},f1={f12},mcc={mcc2}")