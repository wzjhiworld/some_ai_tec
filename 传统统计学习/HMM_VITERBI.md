# Viterbi 算法求解最佳路径

~~~python
import numpy as np

class Node:
    def __init__(self, state):
        self.pre_node = None
        self.state = state

    @staticmethod
    def get_node(state):
        return Node(state)

    def set_pre_node(self, node):
        self.pre_node = node
        return self

h_state_num = 5 # 隐态的数量, 包含一个起始状态和一个终止状态
o_state_num = 4  # 发射态的数量， 0 起始状态固定发射状态， 4 终止状态固定发射状态

h_start_pi = [1.0, 0.0, 0.0, 0.0, 0.0]


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

iter_p = [h_start_pi[i] * h_state_to_o_state[i, o_list[0]]  for i in range(h_state_num)]
iter_p_node = [Node.get_node(i).set_pre_node(None) for i in range(h_state_num)]
iter_p = np.array(iter_p, dtype="float32")

for i in range(1, len(o_list)):
    new_iter_p = np.zeros(len(iter_p))
    new_iter_p_node = [Node.get_node(i) for i in range(h_state_num)]
    for to_h_i in range(h_state_num): # 计算跳转概率
        tmp = h_state_to_h_state[:, to_h_i] * iter_p
        new_iter_p[to_h_i] = np.max(tmp)
        new_iter_p_node[to_h_i].set_pre_node(iter_p_node[np.nanargmax(tmp)])

    for to_h_i in range(h_state_num):
        new_iter_p[to_h_i] = new_iter_p[to_h_i] * h_state_to_o_state[to_h_i][o_list[i]]
    iter_p = new_iter_p
    iter_p_node = new_iter_p_node

index = np.nanargmax(iter_p)
print("prob", iter_p[index])
state_list = []
iter_node = iter_p_node[index]
while iter_node is not None:
    state_list.append(iter_node.state)
    iter_node = iter_node.pre_node

print(state_list[::-1])
~~~
