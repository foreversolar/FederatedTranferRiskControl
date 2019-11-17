#!/usr/bin/env python
# coding: utf-8

# In[12]:


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
import paddlehub as hub
from collections import namedtuple
import codecs
import os
import csv
from paddlehub.dataset import InputExample, HubDataset
from paddlehub.reader.batching import *
import numpy as np


# In[16]:


class MyClassifyReader(hub.reader.ClassifyReader):
    def _pad_batch_records(self, batch_records, phase=None):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            max_seq_len=self.max_seq_len,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids,
            max_seq_len=self.max_seq_len,
            pad_idx=self.pad_id)

        if phase != "predict":
            batch_labels = [record.label_id for record in batch_records]
            batch_labels = np.array(batch_labels).astype("int64").reshape(
                [-1, 1])

            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, batch_labels, batch_seq_lens
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, batch_labels,
                    batch_seq_lens
                ]

        else:
            return_list = [
                padded_token_ids, padded_position_ids, padded_text_type_ids,
                input_mask, batch_seq_lens
            ]

            if self.use_task_id:
                padded_task_ids = np.ones_like(
                    padded_token_ids, dtype="int64") * self.task_id
                return_list = [
                    padded_token_ids, padded_position_ids, padded_text_type_ids,
                    input_mask, padded_task_ids, batch_seq_lens
                ]

        return return_list


# In[14]:


class BiLsTmTextClassifierTask(hub.TextClassifierTask):
    def _build_net(self):
        self.seq_len = fluid.layers.data(
            name="seq_len", shape=[1], dtype='int64')
        seq_len = fluid.layers.assign(self.seq_len)
        
        emb = fluid.layers.sequence_unpad(
            self.feature, length=self.seq_len)
            
        fc0 = fluid.layers.fc(input=emb, size=128 * 4)
        rfc0= fluid.layers.fc(input=emb, size=128 * 4)
        lstm_h,c=fluid.layers.dynamic_lstm(input=fc0,size=128*4,is_reverse=False)
        rlstm_h,c=fluid.layers.dynamic_lstm(input=rfc0,size=128*4,is_reverse=True)
        
        lstm_last=fluid.layers.sequence_last_step(input=lstm_h)
        rlstm_last=fluid.layers.sequence_last_step(input=rlstm_h)
        
        lstm_last_tanh=fluid.layers.tanh(lstm_last)
        rlstm_last_tanh=fluid.layers.tanh(rlstm_last)
        
        lstm_concat = fluid.layers.concat(input=[lstm_last, rlstm_last], axis=1)
        # full connect layer
        fc1 = fluid.layers.fc(input=lstm_concat, size=96, act='tanh')
        
        logits = fluid.layers.fc(
            input=fc1,
            size=self.num_classes,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)),
            act="softmax")

        self.ret_infers = fluid.layers.reshape(
            x=fluid.layers.argmax(logits, axis=1), shape=[-1, 1])

        return [logits]
    
    @property
    def feed_list(self):
        feed_list = [varname for varname in self._base_feed_list]
        if self.is_train_phase or self.is_test_phase:
            feed_list += [self.labels[0].name, self.seq_len.name]
        else:
            feed_list += [self.seq_len.name]
        return feed_list


# In[29]:


class MyDataset(HubDataset):
    """DemoDataset"""
    def __init__(self):
        self.dataset_dir = "./work"
  

    def _load_train_examples(self):
        self.train_file = os.path.join(self.dataset_dir, "QB3.csv")
        self.train_examples = self._read_csv(self.train_file)
    def _load_dev_examples(self):
        self.dev_file = os.path.join(self.dataset_dir, "QBMT.csv")
        self.dev_examples = self._read_csv(self.dev_file)
    

    def _load_test_examples(self):
        self.test_file = os.path.join(self.dataset_dir, "QBMTest.csv")
        self.test_examples = self._read_csv(self.test_file)

    def get_train_examples(self):
        return self.train_examples

    def get_dev_examples(self):
        return self.dev_examples

    def get_test_examples(self):
        return self.test_examples

    def get_labels(self):
        """define it according the real dataset"""
        return ["A轮", "B轮","C轮","天使轮","战略融资"]

    @property
    def num_labels(self):
        """
        Return the number of labels in the dataset.
        """
        return len(self.get_labels())

    def _read_csv(self, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with codecs.open(input_file, "r", encoding="UTF-8") as f:
            reader = csv.reader(f, delimiter=",", quotechar=quotechar)
            examples = []
            seq_id = 0
            header = next(reader)  # skip header
            for line in reader:
                example = InputExample(
                    guid=seq_id, label=line[0], text_a=line[1])
                seq_id += 1
                examples.append(example)
            return examples
ds=MyDataset()


# In[31]:


ds=MyDataset()
module = hub.Module(name="ernie",version="1.1.0")
inputs, outputs, program = module.context(max_seq_len=512)
pooled_output=outputs["sequence_output"]

reader = MyClassifyReader(
    dataset=ds,
    vocab_path=module.get_vocab_path(),
    max_seq_len=512)

strategy=hub.AdamWeightDecayStrategy(
    learning_rate=5e-5,
    lr_scheduler="linear_decay",
    warmup_proportion=0.1,
    weight_decay=0.01,
    optimizer_name="adam"
)
config=hub.RunConfig(use_cuda=False,enable_memory_optim=True,num_epoch=3,batch_size=16,strategy=strategy,checkpoint_dir="BiLSTM_0.38_T3",)

feed_list=[
    inputs["input_ids"].name,inputs["position_ids"].name,
    inputs["segment_ids"].name,inputs["input_mask"].name,
]

data=[["市场需求：国内市场现状：1.公众：创新学习缺少氛围和工具；2.家长/学生：急需创新和个性化的学习方案；3.学校：不知道该如何做创新教育；4.政府：政策大力推动创新收效不明显。业务描述：儿童创新教育服务商，从建筑学出发引导儿童创造，提供课程设计、评估与跟踪体系，并以建筑作为容器和思维方式拓展结合不同学科，设计包括科学、技术、设计、艺术、数学不同学科跨度的创造力思维训练课程。产品介绍：-5-7岁探索力和发散性思维：想象与创造力1、2；空间与角色扮演；牛顿与达芬奇1。-8-10岁架构力和逻辑性思维：设计与实体搭建1、2；博尔赫斯世界；三维建模与3D打印1；牛顿与达芬奇2。-11-12岁科学和文化综合理解：自由机器人；建筑，城市与文明1、2；三维建模与3D打印2。-12岁以上全面、综合的训练学生的创造力：互动与智能；社会创新思维；大型设计与建造。用户画像：5-14岁学生，家长收入来源：1.嘉年华：门票售卖2C，单价100-150；2.套件盒：耗材加盟商供货2B，单价50-150；复用型小教具售卖2C，单价100-500；3.工作坊：教育服务2C，单价200-1W；4.大教具：专用于教学的大教具2B，单价3000-1.5W；孩子自由使用的道具2B，单价1.2-3W；5.加盟商：课程输出/加盟2B，单价4W-20W/年。主攻市场：北京、上海、广州、顺德、淄博、成都、南京和深圳等城市官方网站：http://pacee-edu.com/微信公众号：hellopacee融资需求：未表明融资需求"]]

cls_task=BiLsTmTextClassifierTask(
    data_reader=reader,
    feature=pooled_output,
    feed_list=feed_list,
    num_classes=ds.num_labels,
    config=config
)
map = {3: '天使轮', 1: 'B轮', 4:'战略融资', 0: "A轮", 2: 'C轮'}
predictions=[]
index = 0
run_states = cls_task.predict(data=data)
results = [run_state.run_results for run_state in run_states]
for batch_result in results:
    batch_result = np.argmax(batch_result, axis=2)[0]
    for result in batch_result:
        print(result)
        predictions.append(result)
        index += 1




# In[33]:


result=[]
prob=[]
index=0
for batch_result in results:
    for single_result in batch_result[0]:
        print("=====")
        print(single_result)
        score=(1*single_result[0]+2*single_result[1]+3*single_result[2]+4*single_result[3]+5*single_result[4])/15*100
        print("score:") 
        print(score)


# In[ ]:





# In[ ]:




