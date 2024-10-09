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
from keras.models import load_model
from keras.layers import Dense
import keras
import matplotlib.pyplot as plt
import math


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
log_file_name = "OryzaToArab_set.log"
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

train_set = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(target_train, dtype=tf.float32),
                                                np.asarray(target_train_label).astype('float32').reshape((-1, 1))))
test_set = tf.data.Dataset.from_tensor_slices((tf.convert_to_tensor(target_val, dtype=tf.float32),
                                               np.asarray(target_val_label).astype('float32').reshape((-1, 1))))

num_set=[0.4,0.5,0.6,0.7,0.8]
# num_set=[0.4]
# losses=[]
# xs=[]
for k in range(len(num_set)):
# for i in range(1,21,1):
    loaded_model = load_model('../../../Program3/TFE/step2/modelOA.h5', custom_objects=ak.CUSTOM_OBJECTS)

    model_new = tf.keras.Sequential()
    layers_num = len(loaded_model.layers)
    # 将源域模型逐层加至新模型，并置换最后一层全连接层
    for i in range(layers_num - 1):
        model_new.add(loaded_model.layers[i])
    # print("开始置换最后1层")
    model_new.add(Dense(1, activation='sigmoid', name="l-dense"))  # 置换最后一层全连接层

    # 将模型各层冻结，微调一定的层数
    model_new.trainable = True
    temp = math.ceil(layers_num * num_set[k])
    # temp = math.ceil(layers_num * 0.8)
    for j in range(layers_num - temp):
        model_new.layers[-temp + 1 - j].trainable = False
    #
    # load_model.trainable = False
    # for layer in loaded_model.layers[-4:]:
    #     layer.trainable = True
    #
    model_new.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    model_new.fit(target_train, target_train_label, epochs=100)
    # loaded_model.summary()
    # for i in range(1,30,1):
    # history = model_new.fit(target_train, target_train_label, epochs=500)
    # loss_value = history.history['loss']
    # loss_value = np.array(loss_value)
    # losses.append(loss_value)
    # x = np.array(range(len(loss_value)))
    # xs=x
    # loaded_model.save("trial" + str(k) + ".h5")
    predictions = model_new.predict(target_val)
    accuracy = model_new.evaluate(target_val, target_val_label)
    # loaded_model.save(r"new_model_PAAC.h5")
    print(f"Target Test Accuracy: {accuracy}")
    # loaded_model = load_model("model_3.h5", custom_objects=ak.CUSTOM_OBJECTS)
    # load_model.trainable = False
    # for layer in loaded_model.layers[-4:]:
    #     layer.trainable = True
    # #
    # loaded_model.compile(
    #     optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    #     loss='binary_crossentropy',
    #     metrics=['accuracy']
    # )
    #
    # loaded_model.fit(train_set, epochs=100)
    # loaded_model.save("model_fine_" + str(i)+'.h5')
    # predictions = loaded_model.predict(test_set)

    test_predict1 = predictions
    for j in range(len(predictions)):
        if (float(predictions[j]) > 0.5):
            test_predict1[j] = 1
        else:
            test_predict1[j] = 0
    pre, recall, f1, mcc, acc = evaulate.myEvaulate2(target_val_label, predictions)
    print(f"num_set={num_set[k]},acc={acc},pre={pre},recall={recall},f1={f1},mcc={mcc}")
    # plt.xlabel('epochs', fontsize=14)
    # plt.xticks(fontsize=10)
    # plt.ylabel('loss', fontsize=14)
    # plt.xticks(fontsize=10)
    # plt.plot(x, loss_value)
    # plt.show()
