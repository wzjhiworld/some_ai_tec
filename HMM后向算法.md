# HMM 后向算法实现

~~~python
import numpy as np
import random
h_state_num = 5 # 隐态的数量, 包含一个起始状态和一个终止状态
o_state_num = 4  # 发射态的数量， 0 起始状态固定发射状态， 4 终止状态固定发射状态

h_start_p = [1.0, 0.0, 0.0, 0.0, 0.0]

h_state_to_o_state = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.2, 0.8, 0.0],
    [0.0, 0.6, 0.4, 0.0],
    [0.0, 0.4, 0.6, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])

h_state_to_h_state = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.4, 0.6, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.9, 0.1],
    [0.0, 0.0, 0.0, 0.0, 0.0]
])

o_list = [0, 1, 2, 2, 1] # 观察状态序列

back_iter_p = [h_state_to_o_state[i, o_list[-1]]  for i in range(h_state_num)]
back_iter_p = np.array(back_iter_p, dtype="float32")

for i in range(len(o_list) - 2, -1, -1):
    new_iter_p = np.zeros(len(back_iter_p))
    for to_h_i in range(h_state_num): # 计算跳转概率
        new_iter_p[to_h_i] = np.dot(back_iter_p, h_state_to_h_state[to_h_i, :])

    for to_h_i in range(h_state_num):
        new_iter_p[to_h_i] = new_iter_p[to_h_i] * h_state_to_o_state[to_h_i][o_list[i]]
    back_iter_p = new_iter_p

p_out = np.sum(back_iter_p)

print(p_out) # o_list 发生的完整概率
~~~
