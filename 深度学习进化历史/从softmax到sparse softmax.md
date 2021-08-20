# softmax 到 sparsemax 的进化历史

本文大部分内容借鉴苏神的博客(https://kexue.fm/archives/8046)中的表述，表示感谢。  

## softmax

softmax 主要用于分类问题中，计算每个类别的概率，其公式如下所示:  

<img src="https://latex.codecogs.com/svg.image?p_{i}=\frac{e^{s_{i}}}{\sum_{j=1}^{n}e^{s_{j}}}" title="p_{i}=\frac{e^{s_{i}}}{\sum_{j=1}^{n}e^{s_{j}}}" />  

## softmax 的问题  

分类问题中主要用的loss，主要是交叉熵，公式如下：  

<img src="https://latex.codecogs.com/svg.image?loss&space;=&space;-\hat{p}_{i}logp_{i}" title="loss = -\hat{p}_{i}logp_{i}" />  
  
其中带帽的概率指的是真值，其值为 1.  

我们对上面的loss，我们通过化简得到:  

<img src="https://latex.codecogs.com/svg.image?\begin{align}loss&space;=&space;log(\sum_{j\neq&space;i}^{n}e^{s_{j}-s_{i}}&space;&plus;&space;1)\end{align}" title="\begin{align}loss = log(\sum_{j\neq i}^{n}e^{s_{j}-s_{i}} + 1)\end{align}" />  

假设 <img src="https://latex.codecogs.com/svg.image?s_{j}&space;\;\;\;&space;(j&space;\neq&space;&space;i)" title="s_{j} \;\;\; (j \neq i)" /> 的最小值为<img src="https://latex.codecogs.com/svg.image?s_{min}" title="s_{min}" />, 可以得到:  

<img src="https://latex.codecogs.com/svg.image?\begin{aligned}loss&space;&&space;\geq&space;&space;log(\sum_{j\neq&space;i}^{n}e^{s_{min}-s_{i}}&space;&plus;&space;1)&space;\\&space;&space;&space;&space;&space;&&space;=&space;log((n-1)\cdot&space;e^{s_{min}-s_{i}}&space;&plus;&space;1)&space;&space;\\e^{loss}&space;&&space;\geq&space;(n-1)\cdot&space;e^{s_{min}-s{i}}&space;&plus;&space;1&space;\\e^{s_{min}&space;-&space;s_{i}}&space;&&space;\leq&space;(e^{loss}&space;-&space;1)&space;/&space;(n&space;-&space;1)&space;\\s_{min}&space;-&space;s_{i}&space;&&space;\leq&space;log((e^{loss}&space;-&space;1)&space;/&space;(n&space;-&space;1))&space;\\s_{min}&space;-&space;s_{i}&space;&&space;\leq&space;log(e^{loss}&space;-&space;1)&space;-&space;log(n&space;-&space;1)&space;\\s_{i}&space;-&space;s_{min}&space;&&space;\geq&space;-log(e^{loss}&space;-&space;1)&space;&plus;&space;log(n&space;-&space;1)\end{aligned}" title="\begin{aligned}loss & \geq  log(\sum_{j\neq i}^{n}e^{s_{min}-s_{i}} + 1) \\ & = log((n-1)\cdot e^{s_{min}-s_{i}} + 1) \\e^{loss} & \geq (n-1)\cdot e^{s_{min}-s{i}} + 1 \\e^{s_{min} - s_{i}} & \leq (e^{loss} - 1) / (n - 1) \\s_{min} - s_{i} & \leq log((e^{loss} - 1) / (n - 1)) \\s_{min} - s_{i} & \leq log(e^{loss} - 1) - log(n - 1) \\s_{i} - s_{min} & \geq -log(e^{loss} - 1) + log(n - 1)\end{aligned}" />  

从上面的推到公式可以看出，两个 logit 的差必须要大于log(n-1),这表明分类数越多，网络  
越容易过学习。而实际上分类问题中，我们只需要让 si 大于其他的 sj即可，并不约束一定要  
差距大很多。  

## sparsemax

sparsemax 先将 s1,s2,…,sn 从大到小排列，取其前 k 个 logit 来计算最后的每个类别的  
概率，然后再计算概率。这样做的好处是变向降低了 n，使网络不容易陷入过学习。

## 实验

从苏神的实验中，得出的结论如下：
> 发现它在大多数任务上都有1%的提升，所以非常欢迎大家尝试！不过，我们也发现，Sparse Softmax  
> 只适用于有预训练的场景，因为预训练模型已经训练得很充分了，因此finetune阶段要防止过拟合；但  
> 是如果你是从零训练一个模型，那么Sparse Softmax会造成性能下降，因为每次只有k个类别被学习到，  
> 反而会存在学习不充分的情况（欠拟合）

## 