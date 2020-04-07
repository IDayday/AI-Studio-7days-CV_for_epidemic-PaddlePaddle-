尝试在AI Studio环境下开发与部署深度学习模型，课程由百度飞桨（PaddlePaddle）免费提供。本次课程是为期7天的疫情CV培训。
项目代码包含课程示例和作业内容
### Day01
PaddlePaddle本地安装教程可直接查阅[[官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)]  
PS：注意CPU或GPU版本，请根据系统进行选择。若选择GPU版本，还应配合CUDA以及CUDnn安装。  

在丁香网爬虫疫情数据，并绘制地图。  
使用pyecharts绘制疫情分布图，Pycharts api可参考[[link](https://pyecharts.org/#/zh-cn/)]。Day01/ipynb文件中有demo可供参考。  
以下是截止至2020年3月31日的全国疫情饼图示例  
<div align=center><img src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/sample/%E7%96%AB%E6%83%85%E9%A5%BC%E5%9B%BE%E7%A4%BA%E4%BE%8B.png"/></div>   

### Day02
利用DNN网络识别手势（数字）  
手势图片示例（PS：不知道为什么，看着有点喜感）使用的是AI-Studio平台自带的手势数据集，这个还是很方便的，也可以自定义数据集并上传训练。  
<div align=center><img width="150" height="150" src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/Day02/%E6%89%8B%E5%8A%BF.jpg"/></div>   
课程上给的示例网络是全连接的线性网络（对于初学者比较容易理解），我尝试了一次，效果奇差（avg_acc只有0.12），学习群里有小伙伴各种努力，也只提升到0.5。果断改用CNN完成作业。初次尝试avg_acc就能达到0.93，稍微调整了下优化器和学习率（AdamOpt(lr=0.001）)、训练轮次、FC层节点个数，最终结果达到0.96。V100跑这个项目真的是很轻松。   

### Day03
利用CNN网络识别车牌  
<div align=center><img src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/Day03/%E8%BD%A6%E7%89%8C.png"/></div>  
将上次写的CNN结构改了改，这里输入的尺寸是 128x1x20x20（二值化了原始图片） ，因为图片被处理为单通道（黑白），且进行了分割。所以CNN网络的输入大小也要随之改变。因为输入尺寸里图片大小只有20x20，所以网络没有设计更多层数，pooling也减少（也可以去掉pooling）。最后准确率为0.99。Day03/ipynb内有详细的网络搭建demo。(提示：网络是定义在动态图上，类似于pytorch的网络设计）  

### Day04
利用VGG-16识别人脸口罩  
这次实践降低了难度（单一对象，目标居中），没有要求检测口罩的位置，只需识别。教程给的数据集也比较小，所以VGG-16很容易就在测试集上达到1.0。  
这里VGG-16是自己在动态度上搭建的，当然也可以用Paddlehub内置的api直接加载模型，完成简单测试。  
<div align=center><img src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/Day04/infer_mask01.jpg"/></div>  

## [有待更新]
### Day05
体验Paddlehub and 体验比赛  
- 安装Paddlehub:
  ```
  pip install paddlehub
  ```
- 安装lac模型:
  ```
  hub install lac
  ```
- 执行lac模型预测:
  ```
  hub run lac --input_text"..."
  ```
- 列出本地已安装模型:
  ```
  hub list
  ```
- 用于查看本地已安装模型的属性，包括名字、版本、描述信息等:
  ```
  hub show lac
  ```
- 通过关键字在服务器检索模型:
  ```
  hub search ssd
  ```
- Demo:
  ```
  import paddlehub as hub
  lac = hub.Module(name="lac")
  inputs = {"text":["今天是个好日子"]}
  results = lac.lexical_analysis(data=inputs)
  print(results)
  ```
  ```
  分词结果：['今天'，'是'，'个'，'好日子']
  词性标注：['time','v','q','n']
  ```
最强大的还是迁移学习的能力。
### Day06
PaddleSlim模型压缩及部署（服务于嵌入式设备以及低端芯片）  
模型压缩原理部分在PaddleSlim官方文档中有介绍。[[link](https://paddlepaddle.github.io/PaddleSlim/algo/algo.html)]  
主要分为四个板块：量化、卷积核剪枝、蒸馏、NAS
