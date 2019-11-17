#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import paddle
import paddle.fluid as fluid
import io

# 归一化 
def normalization(data):
    avg = np.mean(data, axis=0)  # axis=0表示按数组元素的列对numpy取相关操作值
    max_ = np.max(data, axis=0)
    min_ = np.min(data, axis=0)
    result_data = (data - avg) / (max_ - min_)
    return result_data

def run_Teampredict(data):
    # #预测
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    # 创建并使用 scope 
    inference_scope = fluid.core.Scope()    
        
    with fluid.scope_guard(inference_scope):
        # 加载预测模型
        path = 'models/Team'
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
        
        t = fluid.LoDTensor()
        a = np.array(data, dtype='float32').reshape(64, 1)
        t.set(a, fluid.CPUPlace())
        t.set_lod([[0, 64]])
        result = exe.run(program=inference_program,
                    feed={feed_target_names[0]: t},
                    fetch_list=fetch_targets)
        print(result[0])
        result = [0 if i<0.42 else 1 for i in result]   #float转int,最后要不要看情况
        return result
        
if __name__ == '__main__':
    alldata = pd.read_csv('./data/alldata.csv')

    alldata = alldata.drop('Company', 1)
    alldata = alldata.drop('Label', 1)
    alldata = np.array(alldata)
    
    alldata = normalization(alldata)    
    
    print(run_Teampredict(alldata[1]))