import paddle.fluid as fluid
import paddle_fl as fl
from paddle_fl.core.master.job_generator import JobGenerator
from paddle_fl.core.strategy.fl_strategy_base import FLStrategyFactory
import math 

class Model(object):
    def __init__(self):
        pass

    def lr_network(self):
        self.hid_dim = 64
        self.hid_dim2 = 8
        self.inputs = fluid.layers.data(name="x1", shape=[1], dtype='float32', lod_level=1)
        self.label = fluid.layers.data(name="y1", shape=[1], dtype='int64')
        
        self.fc0 = fluid.layers.fc(input=self.inputs, size=self.hid_dim)
        self.lstm_h, c = fluid.layers.dynamic_lstm(input=self.fc0, size=self.hid_dim, is_reverse=False)

        # 最大池化
        self.lstm_max = fluid.layers.sequence_pool(input=self.lstm_h, pool_type='max')
        # 激活函数
        self.lstm_max_tanh = fluid.layers.tanh(self.lstm_max)
        # 全连接层
        self.predict = fluid.layers.fc(input=self.lstm_max_tanh, size=self.hid_dim2, act='tanh')
        self.cost = fluid.layers.square_error_cost(input=self.predict, label=self.label)
        self.avg_cost = fluid.layers.mean(x=self.cost)
        
        
        #self.accuracy = fluid.layers.accuracy(input=self.predict, label=self.label)
        #self.loss = fluid.layers.mean(self.sum_cost)
        self.startup_program = fluid.default_startup_program()


model = Model()
model.lr_network()

STEP_EPSILON = 0.1
DELTA = 0.00001
SIGMA = math.sqrt(2.0 * math.log(1.25/DELTA)) / STEP_EPSILON
CLIP = 4.0
batch_size = 64

job_generator = JobGenerator()
optimizer = fluid.optimizer.SGD(learning_rate=0.1)
job_generator.set_optimizer(optimizer)
job_generator.set_losses([model.avg_cost])
job_generator.set_startup_program(model.startup_program)
job_generator.set_infer_feed_and_target_names(
    [model.inputs.name, model.label.name], [model.avg_cost.name])

build_strategy = FLStrategyFactory()
build_strategy.dpsgd = True
build_strategy.inner_step = 1
strategy = build_strategy.create_fl_strategy()
strategy.learning_rate = 0.1
strategy.clip = CLIP
strategy.batch_size = float(batch_size)
strategy.sigma = CLIP * SIGMA

# endpoints will be collected through the cluster
# in this example, we suppose endpoints have been collected
endpoints = ["127.0.0.1:8181"]
output = "fl_job_config"
job_generator.generate_fl_job(
    strategy, server_endpoints=endpoints, worker_num=2, output=output)
# fl_job_config will  be dispatched to workers
