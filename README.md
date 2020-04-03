尝试在AI Studio环境下开发与部署深度学习模型，课程由百度飞桨（PaddlePaddle）免费提供。本次课程是为期7天的疫情CV培训。
项目代码包含课程示例和作业内容
#### Day01
在丁香网爬虫疫情数据，并绘制地图。  
以下是截止至2020年3月31日的全国疫情饼图示例  
<div align=center><img src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/sample/%E7%96%AB%E6%83%85%E9%A5%BC%E5%9B%BE%E7%A4%BA%E4%BE%8B.png"/></div> 
#### Day02
利用DNN网络识别手势（数字）    
手势图片示例（PS：不知道为什么，看着有点喜感）  
<div align=center><img width="150" height="150" src="https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/Day02/%E6%89%8B%E5%8A%BF.jpg"/></div>   
课程上给的示例网络是全连接的线性网络，我尝试了一次，效果奇差（avg_acc只有0.12），学习群里有小伙伴各种努力，也只提升到0.5。果断改用CNN完成作业。初次尝试avg_acc就能达到0.93，稍微调整了下学习率（AdamOpt(lr=0.001）)、训练轮次、FC层节点个数，最终结果达到0.96。
#### Day03
利用CNN网络识别车牌  
![image](https://github.com/IDayday/AI-Studio-7days-CV_for_epidemic/blob/master/Day03/%E8%BD%A6%E7%89%8C.png)  
将上次写的CNN结构改了改，这里输入的尺寸是 128x1x20x20 ，因为图片被处理为单通道（黑白），且进行了分割。所以CNN网络的输入大小也要随之改变。因为输入尺寸里图片大小只有20x20，所以网络没有设计更多层数，pooling也减少。最后准确率为0.99。
