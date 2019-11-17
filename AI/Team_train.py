#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import os
import sys
import math
import paddle
import paddle.fluid as fluid
import io
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


# # 数据处理
alldata = pd.read_csv('./alldata.csv')
# print(len(alldata))

label = alldata['Label'].to_frame()
alldata = alldata.drop('Company', 1)
alldata = alldata.drop('Label', 1)

alldata = np.array(alldata)
label = np.array(label)

# 归一化 
def normalization(data):
    avg = np.mean(data, axis=0)  # axis=0表示按数组元素的列对numpy取相关操作值
    max_ = np.max(data, axis=0)
    min_ = np.min(data, axis=0)
    result_data = (data - avg) / (max_ - min_)
    return result_data

alldata = normalization(alldata)

# Split data into train and test sets as well as for validation and testing
X_train, X_test, Y_train, Y_val = train_test_split(alldata,
                                                   label,
                                                   train_size=0.80,
                                                   random_state=0);

# **构造训练集和测试集的数据生成器**
BATCH_SIZE = 8
BUF_SIZE = 100

# 训练集生成器
def train_generator():
    def reader():
        for i in range(len(X_train)):
            yield [X_train[i], Y_train[i]]
    return reader

# 测试集生成器
def test_generator():
    def reader():
        for i in range(len(X_test)):
            yield [X_test[i], Y_val[i]]
    return reader


# 数据分Batch处理，并打乱减少相关性束缚
train_reader = paddle.batch(
    paddle.reader.shuffle(
        train_generator(),
        buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)

test_reader = paddle.batch(
    paddle.reader.shuffle(
        test_generator(),
        buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)


# In[4]:


# # 建模
# 普通 LSTM网络结构
def lstm_net(input,
             label,
             hid_dim=64,
             hid_dim2=8,
             class_dim = 2):
    
    # Lstm layer
    fc0 = fluid.layers.fc(input=input, size=hid_dim)
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim, is_reverse=False)
    # 最大池化
    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    # 激活函数
    lstm_max_tanh = fluid.layers.tanh(lstm_max)
    # 全连接层
    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2)
    
    prediction = fluid.layers.fc(input=fc1, size=1, act='tanh')
    cost = fluid.layers.square_error_cost(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    # acc = fluid.layers.accuracy(input=prediction, label=label)
    return avg_cost, prediction

# **定义训练过程**
# 1. 定义输入层
# 2. 定义标签层
# 3. 输入层
# 4. 标签层
# 5. 网络结构
# 6. 优化器
# 7. 设备、执行器、feeder定义
# 8. 模型参数初始化
# 9. 双层训练过程
#   *    外层 针对 epoch
#   *    内层 针对 step
#   *    在合适的时机存储参数模型

# 输入层
input = fluid.layers.data(name="x1", shape=[1], dtype='float32', lod_level=1)

# 标签层
label = fluid.layers.data(name="y1", shape=[1], dtype='float32')

# 网络结构
avg_cost, prediction = lstm_net(input, label)

# 优化器
adam_optimizer = fluid.optimizer.Adam(learning_rate=0.001)
adam_optimizer.minimize(avg_cost)

# 设备、执行器、feeder定义
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
feeder = fluid.DataFeeder(feed_list=[input, label], place=place)

# 模型参数初始化
trainer = exe.run(fluid.default_startup_program())

def train(exe,
          train_reader,
          save_dirname,
          pass_num):
    for pass_id in range(pass_num):
        i = 0
        for data in train_reader():
            avg_cost_np, predict = exe.run(fluid.default_main_program(),
                                                    feed=feeder.feed(data),
                                                    fetch_list=[avg_cost, prediction])
            if i % 100 == 0:
                print("Pass {:d}, Batch {:d}, cost {:.6f}".format(pass_id, i, np.mean(avg_cost_np)))
                i += 1
        save_dir = (save_dirname + "/epoch_%d") % pass_id
        paddle.fluid.io.save_inference_model(save_dir, feeded_var_names=['x1'], target_vars=[prediction], executor=exe)
   
    print('train end')


# **训练**
train(exe,
      train_reader,
      save_dirname='./model',
      pass_num=50)


# In[102]:


def test(exe, reader, test_program):
    print('start test')
    result = np.array([])
    y = np.array([])
    for data in reader():
        avg_cost_np, predict, y_label = exe.run(test_program,
                                                feed=feeder.feed(data),
                                                fetch_list=[avg_cost, prediction, label])
        y_label = np.array(y_label).reshape(-1)
        predict = np.array(predict)
        predict = np.mean(predict,axis=1)
        # print(list(y_label),predict)
        y_label = list(y_label)
        predict = [0 if i<0.42 else 1 for i in predict]
        result = np.append(result, predict)
        y = np.append(y, y_label)
        acc = accuracy_score(y_label,predict)
        print(list(y_label),predict)
        print("cost {:.6f}, acc {:.6f}".format(np.mean(avg_cost_np), acc))
    acc = accuracy_score(y,result)
    auc = roc_auc_score(y,result)
    print("total_acc {:.6f}, total_auc {:.6f}".format(acc, auc))
    print('end test')

test_program = fluid.default_main_program().clone(for_test=True)
test(exe,test_reader,test_program)


# In[109]:


# #预测
# 这里要准备预测用的输入数据,格式参考下面的a,这里先用训练数据代替

# 创建并使用 scope 
inference_scope = fluid.core.Scope()    
    
with fluid.scope_guard(inference_scope):
    # 加载预测模型
    path = 'model/epoch_49'
    [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
    
    for data in train_reader():
        t = fluid.LoDTensor()
        a = np.array(data[0][0], dtype='float32').reshape(64, 1)
        t.set(a, fluid.CPUPlace())
        t.set_lod([[0, 64]])
        result = exe.run(program=inference_program,
                    feed={feed_target_names[0]: t},
                    fetch_list=fetch_targets)
        print(result[0])
        result = [0 if i<0.43 else 1 for i in result]   #float转int,最后要不要看情况
        print(result)

