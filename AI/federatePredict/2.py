import pandas as pd 
import numpy as np 
import os
import io
import matplotlib.pyplot as plt
from paddle_fl.core.trainer.fl_trainer import FLTrainerFactory
from paddle_fl.core.master.fl_job import FLRunTimeJob
import numpy
import sys
import paddle
import paddle.fluid as fluid
import logging
import math

logging.basicConfig(filename="test.log", filemode="w", format="%(asctime)s %(name)s:%(levelname)s:%(message)s", datefmt="%d-%M-%Y %H:%M:%S", level=logging.DEBUG)




trainer_id = int(sys.argv[1]) # trainer id for each guest
job_path = "fl_job_config"
job = FLRunTimeJob()
job.load_trainer_job(job_path, trainer_id)
trainer = FLTrainerFactory().create_fl_trainer(job)
trainer.start()

test_program = trainer._main_program.clone(for_test=True)

alldata = pd.read_csv('alldata.csv')
print(len(alldata))


# In[3]:


label = alldata['Label'].to_frame()
alldata = alldata.drop('Company', 1)
alldata = alldata.drop('Label', 1)


# In[4]:


alldata = np.array(alldata)
label = np.array(label)


# In[5]:


# 归一化 
def normalization(data):
    avg = np.mean(data, axis=0)#axis=0表示按数组元素的列对numpy取相关操作值
    max_ = np.max(data, axis=0)
    min_ = np.min(data, axis=0)
    result_data = (data - avg) / (max_ - min_)
    return result_data


# In[6]:


alldata = normalization(alldata)


# In[7]:


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split

# Split data into train and test sets as well as for validation and testing
X_train, X_test, Y_train, Y_val = train_test_split(alldata, 
                                                   label, 
                                                   train_size= 0.80,
                                                   random_state=0);


# **构造训练集和测试集的数据生成器**

# In[17]:


BATCH_SIZE = 8
BUF_SIZE = 100

#训练集生成器
def train_generator():
    def reader():
        for i in range(len(X_train)):
            yield [X_train[i], Y_train[i]]
    return reader

#测试集生成器
def test_generator():
    def reader():
        for i in range(len(X_test)):
            yield [X_test[i], Y_val[i]]
    return reader
    
#数据分Batch处理，并打乱减少相关性束缚
train_reader = paddle.batch(
    paddle.reader.shuffle(
        train_generator(),
        buf_size = BUF_SIZE),
    batch_size = BATCH_SIZE)
    
test_reader = paddle.batch(
    paddle.reader.shuffle(
        test_generator(),
        buf_size = BUF_SIZE),
    batch_size = BATCH_SIZE)

inputs = fluid.layers.data(name="x1", shape=[1], dtype='float32', lod_level=1)
label = fluid.layers.data(name="y1", shape=[1], dtype='float32')
feeder = fluid.DataFeeder(feed_list=[inputs, label], place=fluid.CPUPlace())

def train_test(train_test_program, train_test_feed, train_test_reader):
        cost_set = []
        for test_data in train_test_reader():
            cost = trainer.exe.run(
                program=train_test_program,
                feed=train_test_feed.feed(test_data),
                fetch_list=["mean_0.tmp_0"])
            cost_set.append(float(cost[0]))
        cost_mean = numpy.array(cost_set).mean()
        return cost_mean

def train(train_reader):
    
    #输入层
    data = fluid.layers.data(name="x1", shape=[1], dtype='float32', lod_level=1)
    
    #标签层
    label = fluid.layers.data(name="y1", shape=[1], dtype='float32')
    
    #网络结构
    avg_cost, prediction = network(data, label)

    #优化器
    adam_optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    adam_optimizer.minimize(avg_cost)
    
    #设备、执行器、feeder定义
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)
    
    #模型参数初始化
    exe.run(fluid.default_startup_program())
    
    #双层循环训练
    #外层epoch
    for pass_id in range(pass_num):
        i = 0
        for data in train_reader():
            avg_cost_np= exe.run(fluid.default_main_program(),
                                              feed = feeder.feed(data),
                                              fetch_list=[avg_cost])
            if i % 100 == 0:
                print("Pass {:d}, Batch {:d}, cost {:.6f}".format(pass_id, i, np.mean(avg_cost_np)))
                i += 1
        epoch_model = save_dirname
        fluid.io.save_inference_model(epoch_model, ["x1"], [prediction], exe)
    print('train end')

def compute_privacy_budget(sample_ratio, epsilon, step, delta):
    E = 2 * epsilon * math.sqrt(step * sample_ratio)
    print("({0}, {1})-DP".format(E, delta))

output_folder = "model_node%d" % trainer_id
epoch_id = 0
step = 0
while not trainer.stop():
    epoch_id += 1
    if epoch_id > 40:
        break
    print("epoch %d start train" % (epoch_id))
    for step_id, data in enumerate(train_reader()):
        cost = trainer.run(feeder.feed(data), fetch=["mean_0.tmp_0"])
        step += 1
    print("train cost:%.3f" % (cost[0]))
    
    cost_val = train_test(
        train_test_program=test_program,
        train_test_reader=test_reader,
        train_test_feed=feeder)

    print("Test with epoch %d, cost: %s" % (epoch_id, cost_val))
    compute_privacy_budget(sample_ratio=0.001, epsilon=0.1, step=step, delta=0.00001)
   
    save_dir = (output_folder + "/epoch_%d") % epoch_id
    trainer.save_inference_program(output_folder)


