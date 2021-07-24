# 从 NMS 到 soft-NMS

本文参考的博客

* https://blog.csdn.net/weixin_41665360/article/details/99818073



## 什么是 NMS ？

NMS 中文全称叫做非极大值抑制，主要用于深度学习目标检测的后处理过程中，用于除去冗余的检测框，获得正确的检测结果。

示意图如下：

![20190820194301573](.\images\20190820194301573.png)

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

   ![20190820201119624](.\images\20190820201119624.png)

2. 由于模型训练过程中，分类得到的置信度和回归得到的框坐标并没有直接的相关性，所以置信度高的备选框不一定就在框的位置预测上比置信度低的预测框准确。所以 NMS 以置信度作为第一排序准则也会带来一定的误差。如图：

   ![20190820205242860](.\images\20190820205242860.png)

为了解决上述的问题，有人便发明了 soft-NMS

## soft-NMS

soft-NMS 不在是通过粗暴的将重叠部分大于阈值的检测框直接删掉，而是通过动态的通过重叠的程度，动态减少

检测框的置信度。常用的减少系数函数有线性函数和高斯函数。如下：

<img src="https://latex.codecogs.com/svg.image?score_{i}&space;=&space;\left\{\begin{matrix}score_{i}&space;&&space;iou(M,box_{i})&space;<&space;Threshold&space;\\score_{i}\cdot&space;(1&space;-&space;iou(M,box_{i})&space;&&space;iou(M,box_{i})&space;<&space;Threshold&space;\\\end{matrix}\right.&space;" title="score_{i} = \left\{\begin{matrix}score_{i} & iou(M,box_{i}) < Threshold \\score_{i}\cdot (1 - iou(M,box_{i}) & iou(M,box_{i}) < Threshold \\\end{matrix}\right. " />

关于高斯函数如何使用，大家自行思考。

## soft-NMS 的缺憾

