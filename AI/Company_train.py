#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import KFold
import gc
from collections import Counter
import os
import paddle
import paddle.fluid as fluid


# In[3]:


train= pd.read_csv('train.csv',encoding='utf8')
test= pd.read_csv('data/data12442/test.csv',encoding='utf8')
print(train.shape,test.shape)


# In[4]:


#--------------------两个训练标签--------------------

train_y1=train['label1']
train_y2=train['label2']

train_id=train[['企业名称']]
train=train.drop(['企业名称','label1','label2'],axis=1) #训练数据

submit=pd.DataFrame(list(set(test['企业名称'])),columns=['企业名称']) #待提交企业名称
test_id=test[['企业名称']] #待预测企业名称
test=test.drop(['企业名称'],axis=1) #测试数据

train=np.array(train) #数据类型转换
test=np.array(test) #数据类型转换


# In[5]:


def reader_createor(data, label):
    def reader():
        for i in range(len(data)):
            yield data[i], label[i]
    return reader


# In[6]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Split data into train and test sets as well as for validation and testing
X_train, X_test, Y_train, Y_val = train_test_split(train, 
                                                   np.array(train_y1, dtype='float32'), 
                                                   train_size= 0.80,
                                                   random_state=0);
                                                   


# In[7]:


BUF_SIZE=500
BATCH_SIZE=200

#用于训练的数据提供器，每次从缓存中随机读取批次大小的数据
train_reader = paddle.batch(
    paddle.reader.shuffle(reader=reader_createor(X_train, Y_train), 
                          buf_size=BUF_SIZE),                    
    batch_size=BATCH_SIZE)   
#用于测试的数据提供器，每次从缓存中随机读取批次大小的数据
valid_reader = paddle.batch(
    paddle.reader.shuffle(reader=reader_createor(X_test, Y_val),
                          buf_size=BUF_SIZE),
    batch_size=BATCH_SIZE)  


# In[8]:


# # 建模
# 普通 LSTM网络结构
def lstm_net(input,
             label,
             hid_dim=64,
             hid_dim2=64,
             class_dim = 2):
    
    # Lstm layer
    fc0 = fluid.layers.fc(input=input, size=hid_dim * 4)
    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)
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
            if i % 5 == 0:
                print("Pass {:d}, Batch {:d}, cost {:.6f}".format(pass_id, i, np.mean(avg_cost_np)))
            i += 1
        save_dir = (save_dirname + "/epoch_%d") % pass_id
        paddle.fluid.io.save_inference_model(save_dir, feeded_var_names=['x1'], target_vars=[prediction], executor=exe)
   
    print('train end')


# **训练**
train(exe,
      train_reader,
      save_dirname='./model1',
      pass_num=1)


# In[9]:


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
        predict = [0 if i<0.04 else 1 for i in predict]
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
test(exe,valid_reader,test_program)


# In[15]:


def run_predict(data):
    # #预测
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    # 创建并使用 scope 
    inference_scope = fluid.core.Scope()    
        
    with fluid.scope_guard(inference_scope):
        # 加载预测模型
        path = 'model1/epoch_0'
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
        
        t = fluid.LoDTensor()
        a = np.array(data, dtype='float32').reshape(743, 1)
        t.set(a, fluid.CPUPlace())
        t.set_lod([[0, 743]])
        result = exe.run(program=inference_program,
                    feed={feed_target_names[0]: t},
                    fetch_list=fetch_targets)
        print(result[0])
        result = [0 if i<0.035 else 1 for i in result]   #float转int,最后要不要看情况
        print(result)
        return result
        
if __name__ == '__main__':
    # alldata = pd.read_csv('/home/aistudio/data/data12777/alldata.csv')

    # alldata = alldata.drop('Company', 1)
    # alldata = alldata.drop('Label', 1)
    # alldata = np.array(alldata)
    
    # alldata = normalization(alldata)
    results = []
    for i in range(len(X_test)):
        # if(Y_val[i] == 1):
            results.append(run_predict(X_test[i]))
            print(Y_val[i])
    # run_predict(X_test[348])
    # print(Y_val[348])


# In[16]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

acc = accuracy_score(Y_val,results)
auc = roc_auc_score(Y_val,results)
print("total_acc {:.6f}, total_auc {:.6f}".format(acc, auc))


# In[34]:


count = 0
for i in range(len(Y_val)):
    if(Y_val[i] == 1): 
        count += 1
        print(i)
print(count/len(Y_val))




