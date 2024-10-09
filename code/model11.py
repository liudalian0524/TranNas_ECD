import sys
import os
import autokeras as ak
import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import util.evaulate as evaulate


class Logger(object):
    def __init__(self, file_name="Default.log", stream=sys.stdout):
        self.terminal = stream
        self.log = open(file_name, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


# log_path = './Logs/'
# if not os.path.exists(log_path):
#     os.makedirs(log_path)
# # 日志文件名按照程序运行时间设置model11.py
# log_file_name = "Solanum3.log"
# # 记录正常的 print 信息
# sys.stdout = Logger(log_file_name)
# # 记录 traceback 异常信息
# sys.stderr = Logger(log_file_name)

target_data = pd.read_csv("../../../Data_P/Glycine_Train.csv")
target_pssm = target_data.iloc[:, 1:-1]
target_label = target_data.iloc[:, -1]

Xtrain, Xval, Ytrain, Yval = train_test_split(target_pssm, target_label, test_size=0.7, random_state=300)

train_set = tf.data.Dataset.from_tensor_slices((Xtrain, Ytrain))
test_set = tf.data.Dataset.from_tensor_slices((Xval, Yval))

clf = ak.StructuredDataClassifier(max_trials=10)
clf.fit(train_set, epochs=100)
# model=clf.export_model()
# model.save("model_Solanum.h5")
predictions = clf.predict(test_set)
# # 计算准确率
accuracy = clf.evaluate(test_set)
pre, recall, f1, mcc, acc = evaulate.myEvaulate2(Yval, predictions)
print(f"acc={acc},pre={pre},recall={recall},f1={f1},mcc={mcc}")