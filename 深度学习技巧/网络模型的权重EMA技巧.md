# 网络模型的参数EMA技巧

## 什么是网络模型的权重的EMA技巧 ？

> https://zhuanlan.zhihu.com/p/68748778

EMA技巧是指使用 EMA（指数移动平均）对模型的参数做平均，这样可以提高模型的鲁邦性，
提高泛化能力。  

其中的基本假设是，模型权重在最后的n步内，会在实际的最优点处抖动，所以我们取最后n步 
的平均，使模型更加鲁棒。

## 模拟实验

采用 cifar10 的 mobilenet 分类模型实验来观察效果。

|模型|精度|  
|---|---|  
|no-ema-mobilenetv2|79%|  
|ema-moblienetv2|81%|

从效果上看还是有明显的测试精度提升效果。

代码见同代码仓库目录 ./EMA

