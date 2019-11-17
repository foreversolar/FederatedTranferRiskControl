import pandas as pd
import numpy as np
import paddle
import paddle.fluid as fluid
import io

def run_predict(data):
    # #Ԥ��
    use_cuda = False
    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    # ������ʹ�� scope 
    inference_scope = fluid.core.Scope()    
        
    with fluid.scope_guard(inference_scope):
        # ����Ԥ��ģ��
        path = './models/Company'
        [inference_program, feed_target_names, fetch_targets] = fluid.io.load_inference_model(dirname=path, executor=exe)
        
        t = fluid.LoDTensor()
        a = np.array(data, dtype='float32').reshape(743, 1)
        t.set(a, fluid.CPUPlace())
        t.set_lod([[0, 743]])
        result = exe.run(program=inference_program,
                    feed={feed_target_names[0]: t},
                    fetch_list=fetch_targets)
        print(result[0])
        result = [0 if i<0.05 else 1 for i in result]   #floatתint,���Ҫ��Ҫ�����
        print(result)
        return result
        
if __name__ == '__main__':
    test= pd.read_csv('./data/test.csv',encoding='utf8')

    test_id=test[['��ҵ����']] #��Ԥ����ҵ����
    test=test.drop(['��ҵ����'],axis=1) #��������

    test=np.array(test) #��������ת��
    
    run_predict(test[1])