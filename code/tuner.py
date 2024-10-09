# Copyright 2020 The AutoKeras Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import collections
import copy
import os

import sys
sys.path.append("../..")

import keras_tuner
import tensorflow as tf
from tensorflow import nest
from tensorflow.keras import callbacks as tf_callbacks
from tensorflow.keras.layers.experimental import preprocessing
from sklearn import preprocessing as pp, metrics
from autokeras import pipeline as pipeline_module
from autokeras.utils import data_utils
from autokeras.utils import utils

from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
import math
import matplotlib.pyplot as plt
import torch
from sklearn.metrics import matthews_corrcoef


class AutoTuner(keras_tuner.engine.tuner.Tuner):
    """A Tuner class based on KerasTuner for AutoKeras.

    Different from KerasTuner's Tuner class. AutoTuner's not only tunes the
    Hypermodel which can be directly built into a Keras model, but also the
    preprocessors. Therefore, a HyperGraph stores the overall search space containing
    both the Preprocessors and Hypermodel. For every trial, the HyperGraph build the
    PreprocessGraph and KerasGraph with the provided HyperParameters.

    The AutoTuner uses EarlyStopping for acceleration during the search and fully
    train the model with full epochs and with both training and validation data.
    The fully trained model is the best model to be used by AutoModel.

    # Arguments
        oracle: keras_tuner Oracle.
        hypermodel: keras_tuner HyperModel.
        **kwargs: The args supported by KerasTuner.
    """

    def __init__(self, oracle, hypermodel, **kwargs):
        # Initialize before super() for reload to work.
        self._finished = False
        super().__init__(oracle, hypermodel, **kwargs)
        # Save or load the HyperModel.
        self.hypermodel.save(os.path.join(self.project_dir, "graph"))
        self.hyper_pipeline = None

    def _populate_initial_space(self):
        # Override the function to prevent building the model during initialization.
        return

    def get_best_model(self):
        with keras_tuner.engine.tuner.maybe_distribute(self.distribution_strategy):
            model = tf.keras.models.load_model(self.best_model_path)
        return model

    def get_best_pipeline(self):
        return pipeline_module.load_pipeline(self.best_pipeline_path)

    def _pipeline_path(self, trial_id):
        return os.path.join(self.get_trial_dir(trial_id), "pipeline")

    def _prepare_model_build(self, hp, **kwargs):
        """Prepare for building the Keras model.

        It build the Pipeline from HyperPipeline, transform the dataset to set
        the input shapes and output shapes of the HyperModel.
        """
        dataset = kwargs["x"]
        pipeline = self.hyper_pipeline.build(hp, dataset)
        pipeline.fit(dataset)
        dataset = pipeline.transform(dataset)
        self.hypermodel.set_io_shapes(data_utils.dataset_shape(dataset))

        if "validation_data" in kwargs:
            validation_data = pipeline.transform(kwargs["validation_data"])
        else:
            validation_data = None
        return pipeline, dataset, validation_data

    def _build_and_fit_model(self, trial, *args, **kwargs):
        model = self._try_build(trial.hyperparameters)
        (
            pipeline,
            kwargs["x"],
            kwargs["validation_data"],
        ) = self._prepare_model_build(trial.hyperparameters, **kwargs)
        pipeline.save(self._pipeline_path(trial.trial_id))
        """改动"""
        # print("trail_id",trial.trial_id)

        self.adapt(model, kwargs["x"])

        model, history = utils.fit_with_adaptive_batch_size(
            model, self.hypermodel.batch_size, **kwargs
        )
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print(history.history['loss'])
        print(history.history['val_loss'])

        # loss_value1 =history.history['loss']
        # loss_value1 = np.array(loss_value1)
        # x1 = np.array(range(len(loss_value1)))
        # plt.plot(x1, loss_value1)
        # plt.show()
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("迁移前模型：")
        print(model.layers)
        print("激活函数",model.layers[-1].activation)
        model.summary()
        print("准备返回history")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # model = model.export_model()
        model.save("trial_model_original/trial_model" + str(trial.trial_id) + ".h5")
        layers_num = len(model.layers)  # 模型层数
        print("模型层数为：", layers_num)
        model_new = tf.keras.Sequential()
        # 将源域模型逐层加至新模型，并置换最后一层全连接层
        """改动"""
        for i in range(layers_num-1):
            model_new.add(model.layers[i])
        # print("开始置换最后1层")
        model_new.add(Dense(1, activation='sigmoid',name="l-dense"))  # 置换最后一层全连接层
        """改动"""
        # 将模型各层冻结，微调一定的层数
        tmp = math.ceil(layers_num*0.8)
        model_new.trainable = True
        for j in range(layers_num - tmp):
            model_new.layers[-tmp + 1 - j].trainable = False
        # 激活迁移模型
        """
        这里主要是如何解决目标域数据导入问题
        """
        """
        ##ZT
        model_new.compile(
            optimizer=tf.keras.optimizers.Adadelta(learning_rate=1.0),  # 优化器的选择
            loss='mean_squared_error',  # 损失函数的选择
            metrics=[tf.keras.metrics.mean_squared_error]
        )
        """
        """改动"""
        model_new.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss="binary_crossentropy", ##
            metrics="accuracy"
        )

        target_data = pd.read_csv(r"C:\Users\ldl\PycharmProjects\FNEXM\Data_P_1\HHData_200.csv")
        # target_data = pd.read_csv(r"D:\ZT_keras\Test\data\soucedata.csv")
        x_target_sets = target_data.iloc[:,1:-1]
        y_target_sets = target_data.iloc[:,-1]
        # y_target_sets = np.array(y_target_sets).reshape(-1, 1)


        # x_target_sets = np.loadtxt('D:\\Jupyter_notebook\\JupyterNotebookFile\\datasets\\Transfer_learning\\datatest_2'
        #                            '\\soybeans_data.txt')  # 大豆数据集
        # y_target_sets = np.loadtxt('D:\\Jupyter_notebook\\JupyterNotebookFile\\datasets\\Transfer_learning\\datatest_2'
        #                            '\\soybeans_qy.txt')
        # x_t_train, x_t_test, y_t_train, y_t_test = train_test_split(x_target_sets,
        #                                                             y_target_sets,
        #                                                             test_size=0.3,
        #                                                             random_state=839)  # 对目标域训练集测试集进行划分
        # y_t_train = np.array(y_t_train).reshape(-1, 1)
        # y_t_test = np.array(y_t_test).reshape(-1, 1)
        # x_t_train = pp.RobustScaler().fit_transform(x_t_train)
        # x_t_test = pp.RobustScaler().fit_transform(x_t_test)
        # y_t_train = pp.RobustScaler().fit_transform(y_t_train).ravel()
        # y_t_test = pp.RobustScaler().fit_transform(y_t_test).ravel()



        x_t_train, x_t_test, y_t_train, y_t_test = train_test_split(x_target_sets, y_target_sets,test_size=0.7,random_state=300
                                                                    # ,stratify=y_target_sets
                                                                    )
        # print(x_t_test.shape)
        # x_t_yan,x_t_predict,y_t_yan,y_t_predict = train_test_split(x_t_test,y_t_test,test_size=0.7,random_state=0)
        # print(x_t_predict.shape)
        # enc2 = OneHotEncoder().fit_transform(y_t_train)
        # enc2.toarray()
        # y_t_train = enc2.toarray()
        #
        # enc2 = OneHotEncoder().fit_transform(y_t_test)
        # enc2.toarray()
        # y_t_test = enc2.toarray()

        # y_t_train = np.array(y_t_train)
        history = model_new.fit(
            # x=np.expand_dims(x_t_train, axis=1),
            x=x_t_train,
            y=y_t_train,
            epochs = 300,
            # validation_data = [x_t_test,y_t_test],  #张通师兄的验证集
            validation_split= 0.2  # 我的验证集 总样本 372 , train:test = 3:7; train:111, 其中44用来验证
            # validation_data=[x_t_yan,y_t_yan]
        )


        # self.history = model.fit(state, target_f, epochs=1, batch_size=32）
        # loss_value =history.history['loss']
        # loss_value = np.array(loss_value)
        # x = np.array(range(len(loss_value)))
        # plt.plot(x, loss_value)
        # plt.show()

        # x_t_train = pd.concat([x_t_train,x_t_test])
        # y_t_train = pd.concat([pd.DataFrame(y_t_train),pd.DataFrame(x_t_test)])
        # finetune_model = history.export_model()
        # model_new.save("/model/finetune_model_"+str(trial.trial_id)+".h5")
        print("迁移后模型：")
        # model_new1 = model_new.export_model()
        model_new.save("trial_model/trial_model"+str(trial.trial_id)+".h5")
        # tf.saved_model.save(model_new, )


        y_t_train_predict = model_new.predict(x_t_train) #train+val
        y_t_test_predict = model_new.predict(x_t_test) #test

        print(y_t_train_predict.shape)

        y_t_train_predict1 = y_t_train_predict
        y_t_test_predict1 = y_t_test_predict

        for j in range(len(y_t_train_predict)):
            if (float(y_t_train_predict[j]) > 0.5):
                y_t_train_predict1[j] = 1
            else:
                y_t_train_predict1[j] = 0
        for j in range(len(y_t_test_predict)):
            if (float(y_t_test_predict[j]) > 0.5):
                y_t_test_predict1[j] = 1
            else:
                y_t_test_predict1[j] = 0
        # print(y_t_test_predict1)
        # print(y_t_train_predict1)
        # print(y_t_test_predict)
        # y_t_train_predict = np.argmax(y_t_train_predict, axis=1)
        # y_t_test_predict = np.argmax(y_t_test_predict, axis=1)
        # # print(y_t_test_predict)
        # y_t_train = np.argmax(y_t_train, axis=1)
        # y_t_test = np.argmax(y_t_test, axis=1)
        # print(y_t_test)

        print("\nClassification report for classifier(train):\n\n%s\n"
              % (metrics.classification_report(y_t_train.astype(int), y_t_train_predict1.astype(int),
                                               digits=4)))
        print("Confusion matrix:\n\n%s" % metrics.confusion_matrix(y_t_train.astype(int),
                                                                   y_t_train_predict1.astype(int)))
        print(f"train mcc:{matthews_corrcoef(y_t_train,y_t_train_predict1)}")
        print("\nClassification report for classifier(test):\n\n%s\n"
              % (metrics.classification_report(y_t_test.astype(int), y_t_test_predict1.astype(int), digits=4)))
        print("Confusion matrix:\n\n%s" % metrics.confusion_matrix(y_t_test.astype(int),
                                                                   y_t_test_predict1.astype(int)))
        print(f"test mcc:{matthews_corrcoef(y_t_test,y_t_test_predict1)}")



        print(history.history['loss'])
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print(history.history['val_loss'])
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        return history

    @staticmethod
    def adapt(model, dataset):
        """Adapt the preprocessing layers in the model."""
        # Currently, only support using the original dataset to adapt all the
        # preprocessing layers before the first non-preprocessing layer.
        # TODO: Use PreprocessingStage for preprocessing layers adapt.
        # TODO: Use Keras Tuner for preprocessing layers adapt.
        x = dataset.map(lambda x, y: x)

        def get_output_layers(tensor):
            output_layers = []
            tensor = nest.flatten(tensor)[0]
            for layer in model.layers:
                if isinstance(layer, tf.keras.layers.InputLayer):
                    continue
                input_node = nest.flatten(layer.input)[0]
                if input_node is tensor:
                    if isinstance(layer, preprocessing.PreprocessingLayer):
                        output_layers.append(layer)
            return output_layers

        dq = collections.deque()

        for index, input_node in enumerate(nest.flatten(model.input)):
            in_x = x.map(lambda *args: nest.flatten(args)[index])
            for layer in get_output_layers(input_node):
                dq.append((layer, in_x))

        while len(dq):
            layer, in_x = dq.popleft()
            layer.adapt(in_x)
            out_x = in_x.map(layer)
            for next_layer in get_output_layers(layer.output):
                dq.append((next_layer, out_x))

        return model

    def search(
        self,
        epochs=None,
        callbacks=None,
        validation_split=0,
        verbose=1,
        **fit_kwargs
    ):
        """Search for the best HyperParameters.

        If there is not early-stopping in the callbacks, the early-stopping callback
        is injected to accelerate the search process. At the end of the search, the
        best model will be fully trained with the specified number of epochs.

        # Arguments
            callbacks: A list of callback functions. Defaults to None.
            validation_split: Float.
        """
        print("@@@@@@@@@@@@@@@@@@@@autukeras_engine_tuner@@@@@@@@@@@@@@@@@@@@@@@")
        if self._finished:
            return

        if callbacks is None:
            callbacks = []

        self.hypermodel.set_fit_args(validation_split, epochs=epochs)

        # Insert early-stopping for adaptive number of epochs.
        epochs_provided = True
        if epochs is None:
            epochs_provided = False
            epochs = 1000
            if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
                callbacks.append(
                    tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
                )

        # Insert early-stopping for acceleration.
        early_stopping_inserted = False
        new_callbacks = self._deepcopy_callbacks(callbacks)
        if not utils.contain_instance(callbacks, tf_callbacks.EarlyStopping):
            early_stopping_inserted = True
            new_callbacks.append(
                tf_callbacks.EarlyStopping(patience=10, min_delta=1e-4)
            )

        # Populate initial search space.
        hp = self.oracle.get_space()
        self._prepare_model_build(hp, **fit_kwargs)
        self._try_build(hp)
        print("！！！！！！！！！！！！！！！开始更新搜索空间！！！！！！！！！！！！！！")
        self.oracle.update_space(hp)
        print("!!!!!!!!!!!!!!!!!!!!!!更新搜索空间结束！！！！！！！！！！！！！！！！")
        model = super()    # 这里的super并非model
        print("model:", model)
        model.search(
            epochs=epochs, callbacks=new_callbacks, verbose=verbose, **fit_kwargs
        )
        # print(model.layers)
        print("搜索结束！！！！！！！！！！！！！！！！！！！！！！！！！")
        # Train the best model use validation data.
        # Train the best model with enough number of epochs.
        if validation_split > 0 or early_stopping_inserted:
            copied_fit_kwargs = copy.copy(fit_kwargs)

            # Remove early-stopping since no validation data.
            # Remove early-stopping since it is inserted.
            copied_fit_kwargs["callbacks"] = self._remove_early_stopping(callbacks)

            # Decide the number of epochs.
            copied_fit_kwargs["epochs"] = epochs
            if not epochs_provided:
                copied_fit_kwargs["epochs"] = self._get_best_trial_epochs()

            # Concatenate training and validation data.
            if validation_split > 0:
                copied_fit_kwargs["x"] = copied_fit_kwargs["x"].concatenate(
                    fit_kwargs["validation_data"]
                )
                copied_fit_kwargs.pop("validation_data")

            self.hypermodel.set_fit_args(0, epochs=copied_fit_kwargs["epochs"])
            pipeline, model, history = self.final_fit(**copied_fit_kwargs)
        else:
            # TODO: Add return history functionality in Keras Tuner
            model = self.get_best_models()[0]
            history = None
            pipeline = pipeline_module.load_pipeline(
                self._pipeline_path(self.oracle.get_best_trials(1)[0].trial_id)
            )

        model.save(self.best_model_path)
        pipeline.save(self.best_pipeline_path)
        self._finished = True
        print("@@@@@@@@@@@@@@@@@@@@autukeras_engine_tuner_end@@@@@@@@@@@@@@@@@@@@@@@")
        return history

    def get_state(self):
        state = super().get_state()
        state.update({"finished": self._finished})
        return state

    def set_state(self, state):
        super().set_state(state)
        self._finished = state.get("finished")

    @staticmethod
    def _remove_early_stopping(callbacks):
        return [
            copy.deepcopy(callbacks)
            for callback in callbacks
            if not isinstance(callback, tf_callbacks.EarlyStopping)
        ]

    def _get_best_trial_epochs(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        # steps counts from 0, so epochs = step + 1.
        return self.oracle.get_trial(best_trial.trial_id).best_step + 1

    def _build_best_model(self):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        return self._try_build(best_hp)

    def final_fit(self, **kwargs):
        best_trial = self.oracle.get_best_trials(1)[0]
        best_hp = best_trial.hyperparameters
        pipeline, kwargs["x"], kwargs["validation_data"] = self._prepare_model_build(
            best_hp, **kwargs
        )

        model = self._build_best_model()
        self.adapt(model, kwargs["x"])
        model, history = utils.fit_with_adaptive_batch_size(
            model, self.hypermodel.batch_size, **kwargs
        )
        return pipeline, model, history

    @property
    def best_model_path(self):
        return os.path.join(self.project_dir, "best_model")

    @property
    def best_pipeline_path(self):
        return os.path.join(self.project_dir, "best_pipeline")

    @property
    def objective(self):
        return self.oracle.objective

    @property
    def max_trials(self):
        return self.oracle.max_trials
