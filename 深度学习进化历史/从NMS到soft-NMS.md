# 从 NMS 到 soft-NMS

本文参考的博客

* https://blog.csdn.net/weixin_41665360/article/details/99818073



## 什么是 NMS ？

NMS 中文全称叫做非极大值抑制，主要用于深度学习目标检测的后处理过程中，用于除去冗余的检测框，获得正确的检测结果。

示意图如下：

![20190820194301573](https://user-images.githubusercontent.com/78289886/126869656-6d545ba1-4a30-43af-8cf8-113ab9ca5d8e.png)

算法过程如下：

1. 将网络输出的检测框 Box 按照置信度分数 Score 从高到低进行排序，得到检测框集合 Boxes， 设最终的检测框集合为 Dest

2. 当 Boxes 不为空集时：

   1. 从 Boxes 中取出当前置信度最高的检测框 box，将其放入目标集合 Dest 中，并将其从 Boxes 中删除。

   2. 对于 Boxes 中余下的每个框 box_i:

      如果 iou(m, bi) >=  Threshold (阈值)，则将 box_i 从 Boxes集合中删除，返回第二大步。

    3. 返回检出集合 Dest

**解释：**

NMS 算法的核心思想就是将所有的检出框中与已经存在的置信度较高的框重叠过多的框进行剔除。

## NMS 的问题

NMS 的典型问题：

1. 稠密场景下漏检较多(两个或多个目标挨得过近)，如果两个目标存在部分重叠时，可能会漏检。如图:

![20190820201119624](https://user-images.githubusercontent.com/78289886/126869661-ad343926-4b74-4fec-bf27-fdbf89b3c9f8.png)

2. 由于模型训练过程中，分类得到的置信度和回归得到的框坐标并没有直接的相关性，所以置信度高的备选框不一定就在框的位置预测上比置信度低的预测框准确。所以 NMS 以置信度作为第一排序准则也会带来一定的误差。如图：

![20190820205242860](https://user-images.githubusercontent.com/78289886/126869664-e4be35b8-12c4-4297-915e-899777c06147.png)

为了解决上述的问题，有人便发明了 soft-NMS

## soft-NMS

soft-NMS 不在是通过粗暴的将重叠部分大于阈值的检测框直接删掉，而是通过动态的通过重叠的程度，动态减少

检测框的置信度。常用的减少系数函数有线性函数和高斯函数。如下：

<img src="https://latex.codecogs.com/svg.image?score_{i}&space;=&space;\left\{\begin{matrix}score_{i}&space;&&space;iou(M,box_{i})&space;<&space;Threshold&space;\\score_{i}\cdot&space;(1&space;-&space;iou(M,box_{i})&space;&&space;iou(M,box_{i})&space;<&space;Threshold&space;\\\end{matrix}\right.&space;" title="score_{i} = \left\{\begin{matrix}score_{i} & iou(M,box_{i}) < Threshold \\score_{i}\cdot (1 - iou(M,box_{i}) & iou(M,box_{i}) < Threshold \\\end{matrix}\right. " />

关于高斯函数如何使用，大家自行思考。

## soft-NMS 的缺憾

soft-NMS 一定程度上缓解了稠密场景的问题，但是对预测位置框的精确问题没有进行解决。后续诞生了

softer-NMS 对最终输出的位置的建模进行了处理。

## softer-NMS

softer-NMS 在 soft-NMS 的基础上，加入了位置置信度的概念（实际上就是高斯分布的方差，预测准确方差小，预测不准确方差大）。在网络模型的输出中，不仅需要预测框的 4 个位置参数，还需要预测每个位置的置信度。

其推到过程主要是首先假设输出的框的 4 个位置坐标相互独立，并满足高斯分布:

<img src="https://latex.codecogs.com/svg.image?P_{\Theta&space;}(x)=\frac{1}{\sqrt{2\pi&space;\sigma^{2}}}\cdot&space;e^{-\frac{(x-x_{e})^{2}}{2\cdot\sigma^{2}}}" title="P_{\Theta }(x)=\frac{1}{\sqrt{2\pi \sigma^{2}}}\cdot e^{-\frac{(x-x_{g})^{2}}{2\cdot\sigma^{2}}}" />

其中<img src="https://latex.codecogs.com/svg.image?x_{e}" title="x_{g}" />为需要预测的标注框，<img src="https://latex.codecogs.com/svg.image?\sigma^{2}&space;" title="\sigma^{2} " />是预测的模型的方差，如果预测的越准，则方差越小，预测的越不准则方差越大。

同时假设真实的 Ground 框的分布也是高斯分布，只是是处于方差<img src="https://latex.codecogs.com/svg.image?\sigma^{2}&space;\to&space;0" title="\sigma^{2} \to 0" />的时候，这时候，高斯分布退化为 Dirac delta 函数。

<img src="https://latex.codecogs.com/svg.image?P_{G}(x)&space;=&space;\delta&space;(x&space;-&space;x_{g})" title="P_{G}(x) = \delta (x - x_{g})" />

(还记得以前学习的冲击响应吗？)

最后将 预测的概率分布 与 真实的概率分布求KL散度作为最终的损失函数：

<img src="https://latex.codecogs.com/svg.image?KL(P_{G}(x)||P_{\Theta&space;}(x))&space;=&space;\frac{(x_{g}-x_{e})^{2}}{2\sigma^{2}}&space;&plus;&space;\frac{log(\sigma&space;^{2})}{2}&space;&plus;&space;\frac{log(2\pi&space;)}{2}&space;-&space;H(P_{G}(x))" title="KL(P_{G}(x)||P_{\Theta }(x)) = \frac{(x_{g}-x_{e})^{2}}{2\sigma^{2}} + \frac{log(\sigma ^{2})}{2} + \frac{log(2\pi )}{2} - H(P_{G}(x))" />

其中 H 函数为求熵函数。从上式可以看出，KL 散度只与公式右边的前两项有关，后两项都是常数。从公式中可以看出如果要将 KL 散度最小化，如果预测的特别准（<img src="https://latex.codecogs.com/svg.image?x_{g}-x_{e}\to&space;0" title="x_{g}-x_{e}\to 0" />），这个时候，第二项中的方差就可以预测得更小，表示预测得更精确，如果预测的不准，则可以通过输出一个较大的方差来减少KL散度的值。这样模型不仅给出了相关的位置的预测，还同时给出了预测是否精准的一个评估参数<img src="https://latex.codecogs.com/svg.image?\sigma^{2}&space;" title="\sigma^{2} " />。

> KL 散度的公式在计算的过程中可能也存在一些问题，例如当<img src="https://latex.codecogs.com/svg.image?\sigma^{2}&space;\to&space;0" title="\sigma^{2} \to 0" />时，公式中的第二项需要是一个绝对值特别大的负数，可能会导致一些数值计算精度上的问题。公式中的第一项也存在同样的问题。所以感觉可以通过加入一个小的常系数来缓解数值计算带来的训练过程中的问题。

通过上述的模型输出方式，我们就得到了关于位置准确度的一个评估，可以用于矫正框回归的位置的准确性。softer-NMS 的算法流程如下：

![20190823153138458](E:\Github\some_ai_tec\深度学习进化历史\images\20190823153138458.png)

其中位置框的更新规则如下：

<img src="https://latex.codecogs.com/svg.image?\begin{equation}\begin{aligned}p_{i}&space;&=&space;e^{-\frac{(1-iou(bi,b))^{2}}{\sigma_{t}}}&space;\\x&space;&=&space;\frac{\sum_{i}^{}p_{i}x_{i}/\sigma_{x,i}^{2}}{\sum_{i}^{}p_{i}/\sigma_{x,i}^{2}}&space;\\&&space;subject&space;\&space;to&space;\&space;&space;iou(b_{i},&space;b)&space;>&space;0\end{aligned}\end{equation}&space;" title="\begin{equation}\begin{aligned}p_{i} &= e^{-\frac{(1-iou(bi,b))^{2}}{\sigma_{t}}} \\x &= \frac{\sum_{i}^{}p_{i}x_{i}/\sigma_{x,i}^{2}}{\sum_{i}^{}p_{i}/\sigma_{x,i}^{2}} \\& subject \ to \ iou(b_{i}, b) > 0\end{aligned}\end{equation} " />

从上可以看出，通过iou和方差给每个预测的位置参数都进行了加权，可以拟合得到更好的框位置参数。

## 总结

本文阐述了从 NMS 到 soft-NMS 到 softer-NMS 的发展，从中可以看出，目标检测后处理的发展概况，及其优化方向。

1. NMS 只是剔除冗余检测框
2. soft-NMS 通过动态的压制重叠区域的阈值缓解稠密问题
3. softer-NMS 通过对位置的概率建模,得以有效聚合不同位置框的参数，提升位置框预测的准确性。
