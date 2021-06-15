# HMM 前向算法

**代码示例**

~~~python
import numpy as np
import random
h_state_num = 5 # 隐态的数量, 包含一个起始状态和一个终止状态
o_state_num = 4  # 发射态的数量， 0 起始状态固定发射状态， 4 终止状态固定发射状态

# h_start_pi = [ random.random() for _ in range(h_state_num) ]  #起始隐态
h_start_pi = [1.0, 0.0, 0.0, 0.0, 0.0]
h_start_pi = np.asarray(h_start_pi, dtype="float32") / np.sum(h_start_pi)

# h_state_to_o_state = np.random.randn(h_state_num, o_state_num)
h_state_to_o_state = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 0.2, 0.8, 0.0],
    [0.0, 0.6, 0.4, 0.0],
    [0.0, 0.4, 0.6, 0.0],
    [0.0, 0.0, 0.0, 1.0]
])
# h_state_to_o_state = np.array(h_state_to_o_state) / np.sum(h_state_to_o_state, axis = 1).reshape(-1, 1)

# h_state_to_h_state = np.random.randn(h_state_num, h_state_num)
h_state_to_h_state = np.array([
    [0.0, 1.0, 0.0, 0.0, 0.0],
    [0.0, 0.4, 0.6, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5, 0.0],
    [0.0, 0.0, 0.0, 0.9, 0.1],
    [0.0, 0.0, 0.0, 0.0, 0.0]
])
# h_state_to_h_state = np.array(h_state_to_h_state) / np.sum(h_state_to_h_state, axis = 1).reshape(-1, 1)

o_list = [0, 1, 2, 2, 1, 3] # 观察状态序列

iter_p = [h_start_pi[i] * h_state_to_o_state[i, o_list[0]]  for i in range(h_state_num)]
iter_p = np.array(iter_p, dtype="float32")

for i in range(1, len(o_list)):
    new_iter_p = np.zeros(len(iter_p))
    for to_h_i in range(h_state_num): # 计算跳转概率
        new_iter_p[to_h_i] = np.dot(iter_p, h_state_to_h_state[:, to_h_i])

    for to_h_i in range(h_state_num):
        new_iter_p[to_h_i] = new_iter_p[to_h_i] * h_state_to_o_state[to_h_i][o_list[i]]
    iter_p = new_iter_p

p_out = np.sum(iter_p)

print(p_out) # o_list 发生的完整概率
~~~
