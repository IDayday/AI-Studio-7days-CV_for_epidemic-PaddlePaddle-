尝试在AI Studio环境下开发与部署深度学习模型，课程由百度飞桨（PaddlePaddle）免费提供。本次课程是为期7天的疫情CV培训。
项目代码包含课程示例和作业内容
### Day01
PaddlePaddle本地安装教程可直接查阅[[官网](https://www.paddlepaddle.org.cn/documentation/docs/zh/install/index_cn.html)]  
- 这里介绍我在win10平台下安装GPU版本的过程：
```
为了方便管理，我使用了Anaconda进行管理
```
  - 第一步：新建环境
  ```
  conda creat -n paddle python=3.6
  ```
  - 第二步：检查CUDA以及CUDnn安装情况
  ```
  cmd: nvcc -V
  ```
  - 第三步：安装
  ```
  conda: conda activate paddle(进入环境)
  conda: conda install paddlepaddle-gpu cudatoolkit=9.0(我的CUDA是9.0)
  ```
  - 第四步：验证安装
  ```
  conda: conda activate paddle(进入环境)
  conda: python(进入python解释器)
  python: import paddle.fluid
          paddle.fluid.install_check.run_check()
  出现：Your Paddle Fluid is installed succesfully!说明安装成功
  ```
PS：注意CPU或GPU版本，请根据系统进行选择。若选择GPU版本，还应配合对应版本CUDA以及CUDnn安装。  

在丁香网爬虫疫情数据，并绘制地图。  
使用pyecharts绘制疫情分布图，Pycharts api可参考[[link](https://pyecharts.org/#/zh-cn/)]。Day01/ipynb文件中有demo可供参考。  
- 绘图的主要api如下所示：
```
# 地图绘制
m = Map()
m.add("累计确诊", [list(z) for z in zip(labels, counts)], 'china')
#系列配置项,可配置图元样式、文字样式、标签样式、点线样式等
m.set_series_opts(label_opts=opts.LabelOpts(font_size=12),
                  is_show=False)
#全局配置项,可配置标题、动画、坐标轴、图例等
m.set_global_opts(title_opts=opts.TitleOpts(title='全国实时确诊数据',
                                            subtitle='数据来源：丁香园'),
                  legend_opts=opts.LegendOpts(is_show=False),
                  visualmap_opts=opts.VisualMapOpts(pieces=pieces,
                                                    is_piecewise=True,   #是否为分段型
                                                    is_show=True))       #是否显示视觉映射配置
```
```
# 饼图绘制
p = Pie()
p.add("", [list(z) for z in zip(labels, counts)],center=[280,350],radius=[5,100])
#系列配置项,可配置图元样式、文字样式、标签样式、点线样式等
p.set_series_opts(label_opts=opts.LabelOpts(formatter="{b}: {c}",font_size=12),
                  is_show=True)
#全局配置项,可配置标题、动画、坐标轴、图例等
p.set_global_opts(title_opts=opts.TitleOpts(title='全国疫情饼图示例',
                                            subtitle='数据来源：丁香园'),
                  legend_opts=opts.LegendOpts(is_show=False))
 ```

以下是截止至2020年3月31日的全国疫情饼图示例  
<div align=center><img src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/sample/%E7%96%AB%E6%83%85%E9%A5%BC%E5%9B%BE%E7%A4%BA%E4%BE%8B.png"/></div>   

### Day02
利用DNN网络识别手势（数字）  
手势图片示例（PS：不知道为什么，看着有点喜感）使用的是AI-Studio平台自带的手势数据集，这个还是很方便的，也可以自定义数据集并上传训练。  
<div align=center><img width="150" height="150" src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/Day02/%E6%89%8B%E5%8A%BF.jpg"/></div>   
课程上给的示例网络是全连接的线性网络（对于初学者比较容易理解），我尝试了一次，效果奇差（avg_acc只有0.12），学习群里有小伙伴各种努力，也只提升到0.5。果断改用CNN完成作业。初次尝试avg_acc就能达到0.93，稍微调整了下优化器和学习率（AdamOpt(lr=0.001）)、训练轮次、FC层节点个数，最终结果达到0.96。V100跑这个项目真的是很轻松。  

- paddle读取数据需要定义一个reader来喂数据给模型:
```
# 用于训练的数据提供器
train_reader = paddle.batch(reader=paddle.reader.shuffle(reader=data_reader('./train_data.list'), buf_size=256), batch_size=32)
# 用于测试的数据提供器
test_reader = paddle.batch(reader=data_reader('./test_data.list'), batch_size=32) 
```
- 其中 data_reader定义如下：
```
def data_reader(data_list_path):
    def reader():
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                img, label = line.split('\t')
                yield img, int(label)
    return paddle.reader.xmap_readers(data_mapper, reader, cpu_count(), 512)
```
- 自定义网络结构以及forward方式：(类似于pytorch的语法)
```
#定义VGG网络(动态图)
class VGG(fluid.dygraph.Layer):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv1 = Conv2D(3, 16, 3, act='relu')
        self.pool1 = Pool2D(pool_size=2,pool_type='max',pool_stride=2)
        self.conv2 = Conv2D(16, 32, 3, act='relu')
        self.pool2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2)
        self.conv3 = Conv2D(32, 64, 3, act='relu')
        self.pool3 = Pool2D(pool_size=2,pool_type='max',pool_stride=2)
        self.conv4 = Conv2D(64, 128, 3, act='relu')
        self.conv5 = Conv2D(128, 128, 3, act='relu')
        self.linear1 = Linear(input_dim=4608, output_dim=2000, act='relu')# 6*6*128
        self.predict = Linear(input_dim=2000, output_dim=10, act='softmax')

# 网络的前向计算过程
    def forward(self, input):
        x = self.conv1(input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = fluid.layers.reshape(x, shape=[-1,4608])
        x = self.linear1(x)
        y = self.predict(x)
        return y
```

### Day03
利用CNN网络识别车牌  
<div align=center><img src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/Day03/%E8%BD%A6%E7%89%8C.png"/></div>  
将上次写的CNN结构改了改，这里输入的尺寸是 128x1x20x20（二值化了原始图片） ，因为图片被处理为单通道（黑白），且进行了分割。所以CNN网络的输入大小也要随之改变。因为输入尺寸里图片大小只有20x20，所以网络没有设计更多层数，pooling也减少（也可以去掉pooling）。最后准确率为0.99。Day03/ipynb内有详细的网络搭建demo。(提示：网络是定义在动态图上，类似于pytorch的网络设计）  

- 简单修改之前定义的网络结构(注释里有详细的维度变换):   
```
#定义VGGNET网络
class VGGNET(fluid.dygraph.Layer):
    def __init__(self):
        super(VGGNET, self).__init__()
        self.conv1 = Conv2D(1, 16, 3,act='relu')# 128 16 18 18
        self.conv2 = Conv2D(16, 32, 3,act='relu')# 128 32 16 16
        self.pool1 = Pool2D(pool_size=2,pool_type='max',pool_stride=2)# 128 32 8 8
        self.conv3 = Conv2D(32, 64, 3,act='relu')# 128 64 6 6
        self.conv4 = Conv2D(64, 128, 3, act='relu')# 128 128 4 4
        self.pool2 = Pool2D(pool_size=2,pool_type='max',pool_stride=2)# 128 128 2 2
        self.linear1 = Linear(input_dim=512, output_dim=256, act='relu')# 128*2*2
        self.drop_ratio = 0.5
        self.predict = Linear(input_dim=256, output_dim=65, act='softmax')
# 网络的前向计算过程
    def forward(self, input):
        x = self.conv1(input)
        x = self.conv2(x)
        x = self.pool1(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool2(x)
        x = fluid.layers.reshape(x, [x.shape[0],-1])
        x = self.linear1(x)
        x = fluid.layers.dropout(x, self.drop_ratio)
        y = self.predict(x)
        return y
```  
- 在训练中可以设置自动更新学习速率：  
```
auto_rate = paddle.fluid.layers.piecewise_decay([100,200],[0.001,0.0005,0.0001])
opt=fluid.optimizer.AdamOptimizer(learning_rate=auto_rate, parameter_list=model.parameters())
```  

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
<div align=center><img src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic-PaddlePaddle-/blob/master/sample/PaddleSlim.jpg"/></div>  

- 重点介绍卷积通道裁剪:
  ```
  定义：裁剪掉不重要的冗余的卷积参数
  目的：减少参数量，加快推理速度
  ```

首先评估参数的重要性的两种方法：1.敏感度评估卷积层整体的重要性。2.L1_norm评估卷积层内通道的重要性。  

方法1：基于在测试集上的敏感度确定每个卷积层剪裁比例。  

<div align=center><img src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic-PaddlePaddle-/blob/master/sample/%E6%95%8F%E6%84%9F%E5%BA%A6%E5%88%86%E6%9E%90.jpg"/></div>  

方法2：单个卷积内，按通道的L1_norm进行排序。  
裁剪结果一般是：FLOPs大量下降，精度略有浮动，速度显著提升。  

  
