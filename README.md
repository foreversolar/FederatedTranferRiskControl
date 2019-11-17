千信智服-基于联邦学习的金融风控平台

# 功能介绍

​		千信智服是一个金融科技服务平台，利用联邦学习和迁移学习在金融风控领域的优势，打破数据孤岛、保障数据安全。在实现数据隔离的前提下，通过多方数据聚合设计评估体系，自动化评估流程，评估初创企业的成长风险，帮助金融机构识别企业的潜在风险，有效管理和降低风险可能带来的损失。

​        千信智服基于金融研究机构、第三方企业、互联网公开数据集，结合PADDLEPADDLE、PADDLEHUB、AIStudio进行应用创新，结合百度开放API，舆论分析、知识图谱实体标注、预训练模型库，实现核心算法应用：联邦学习共建模型、产品介绍智能分类、团队能力预测、企业能力预测等，最终平台呈现行情汇报、风险预测、风险跟踪、投资决策、账号管理等各项服务能力。

# 运行环境

Ubuntu 16.04

Python 3.7

Paddlepaddle 1.6/1.5（Product_predict）

PaddleFL 0.1

Paddlehub 1.1.1

sklearn 

# 工程结构介绍

- AI
  - data：训练及测试数据
  - models：部分预训练模型
  - **_predict.py: 预测文件
  - **_train.py:训练文件
- Server
- FrontPage

# 详细文件介绍

## **FederatePredict**

**User-Defined-Program**: PaddlePaddle的程序定义了机器学习模型结构和训练策略

**Distributed-Config**: 在联邦学习中，系统会部署在分布式环境中。分布式训练配置定义分布式训练节点信息。

**FL-Job-Generator**: 给定FL-Strategy, User-Defined Program 和 Distributed Training Config，联邦参数的Server端和Worker端的FL-Job将通过FL Job Generator生成。FL-Jobs 被发送到组织和联邦参数服务器以进行联合训练。
**fltrainer.py** ：喂入数据，联邦各方训练
**flserver.py**  ：服务端，云或第三方集群中运行的联邦参数服务器。
**run.sh** ：运行整个项目

## ProductPredict

**Product_predict.py** : 预测文件，直接运行即可。模型需要提前从网盘中下载，环境需要是paddle1.5(1.6还没搞懂为什么不行，貌似你们改了接口。)

**Product_train.py**：数据集经过百度API的知识图谱实体标注进行过处理，使用ERNIE+BiLstm/GRU进行finetune训练。

**models/Product**：训练好的模型

## TeamPredict

**Team_predict.py** : 预测文件，直接运行即可。模型需要提前从网盘中下载，环境需要是paddle1.5(1.6还没搞懂为什么不行，貌似你们改了接口。)

**Team_train.py**：数据集经过百度API的知识图谱实体标注进行过处理，使用ERNIE+BiLstm/GRU进行finetune训练。

**models/Product**：训练好的模型

## CompanyPredict

# 项目地址集合

## AISTUDIO

千信智服-产品分析：<https://aistudio.baidu.com/aistudio/projectdetail/125605>

## 产品网页地址

## 模型地址

models/Product: 放入：https://pan.baidu.com/s/1xUjlnnukI_QXb-Q5V7Xi4g