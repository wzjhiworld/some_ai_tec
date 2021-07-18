# HMM 训练算法代码

~~~python
import numpy as np
from sklearn import preprocessing
hide_state_num = 3
output_state_num = 3
sample_length = 3  # 单个样本的时间布度
epsilon = 1e-5

hide_state_start  = preprocessing.normalize(np.random.random((1, hide_state_num)), "l1").reshape(-1)
hide_state_to_hide_state = preprocessing.normalize(np.random.random((hide_state_num, hide_state_num)), "l1")
hide_state_to_output_state = preprocessing.normalize(np.random.random((hide_state_num, output_state_num)), "l1")


def calcu_forward_matrix(sample_output,
                         hide_state_start = hide_state_start,
                         hide_state_to_hide_state = hide_state_to_hide_state,
                         hide_state_to_output_state = hide_state_to_output_state):
    """
    计算前向状态矩阵
    :param sample_output:
    :return:
    """
    forward_matrix = np.zeros((hide_state_num, sample_length), dtype="float32")
    forward_matrix[0] = hide_state_start * hide_state_to_output_state[:, sample_output[0]]
    for i in range(1, sample_length):
        forward_matrix[i] = np.dot(forward_matrix[i - 1].reshape((1, hide_state_num)), hide_state_to_hide_state)[0]
        forward_matrix[i] = forward_matrix[i] * hide_state_to_output_state[:, sample_output[i]]
    return forward_matrix

# print(calcu_forward_matrix([0, 1, 2, 3]))

def calcu_backward_matrix(sample_output,
                          hide_state_start = hide_state_start,
                          hide_state_to_hide_state = hide_state_to_hide_state,
                          hide_state_to_output_state = hide_state_to_output_state):
    """
    计算后向状态矩阵
    :param sample_output:
    :return:
    """
    backward_matrix = np.zeros((hide_state_num, sample_length), dtype="float32")
    backward_matrix[sample_length - 1] = hide_state_to_output_state[:, sample_output[sample_length - 1]]
    for i in range(sample_length - 2, -1, -1):
        tmp = np.dot(hide_state_to_hide_state, backward_matrix[i + 1].reshape(hide_state_num, 1))[:, 0]
        backward_matrix[i] = tmp * hide_state_to_output_state[:, sample_output[i]]
    return backward_matrix

# print(calcu_backward_matrix([0, 1, 2, 3]))

def iter_func(sample_output_list,
             hide_state_start,
             hide_state_to_hide_state,
             hide_state_to_output_state):
    new_hide_state_start = np.zeros(hide_state_num, dtype="float32")
    new_hide_state_to_hide_state = np.zeros((hide_state_num, hide_state_num), dtype="float32")
    new_hide_state_to_output_state = np.zeros((hide_state_num, output_state_num), dtype="float32")

    forward_matrix_list = [calcu_forward_matrix(sample, hide_state_start, hide_state_to_hide_state, hide_state_to_output_state) for sample in sample_output_list]
    backward_matrix_list = [calcu_backward_matrix(sample, hide_state_start, hide_state_to_hide_state, hide_state_to_output_state) for sample in sample_output_list]

    for hide_state_i in range(hide_state_num):
        a = np.sum([forward_matrix_list[sample_i][0][hide_state_i] * np.dot(backward_matrix_list[sample_i][1], hide_state_to_hide_state[hide_state_i, :]) for sample_i in range(len(sample_output_list))])
        b = np.sum([np.sum(forward_matrix_list[sample_i][-1]) for sample_i in range(len(sample_output_list))])
        new_hide_state_start[hide_state_i] = (a + epsilon) / (b + epsilon * hide_state_num)

    for hide_state_i in range(hide_state_num):
        for hide_state_j in range(hide_state_num):
            a = np.sum([np.sum([forward_matrix_list[sample_i][t_i][hide_state_i] * hide_state_to_hide_state[hide_state_i][hide_state_j] * backward_matrix_list[sample_i][t_i + 1][hide_state_j] for t_i in range(sample_length - 1)]) for sample_i in range(len(sample_output_list))])
            b = np.sum([np.sum([forward_matrix_list[sample_i][t_i][hide_state_i] * np.dot(hide_state_to_hide_state[hide_state_i].reshape(-1, hide_state_num), backward_matrix_list[sample_i][t_i + 1].reshape(hide_state_num, -1))[0][0] for t_i in range(sample_length - 1)]) for sample_i in range(len(sample_output_list))])
            new_hide_state_to_hide_state[hide_state_i][hide_state_j] = (a + epsilon) / (b + hide_state_num * epsilon)

    def is_ok_output(o1, o2):
        if o1 == o2:
            return 1.0
        else:
            return 0.0

    for hide_state_i in range(hide_state_num):
        for output_state_j in range(output_state_num):
            a = np.sum([forward_matrix_list[sample_i][t_i][hide_state_i] * np.dot(backward_matrix_list[sample_i][t_i + 1], hide_state_to_hide_state[hide_state_i, :]) * is_ok_output(sample_output_list[sample_i][t_i], output_state_j) for t_i in range(sample_length - 1) for sample_i in range(len(sample_output_list))])
            a += np.sum([forward_matrix_list[sample_i][sample_length - 1][hide_state_i] * is_ok_output(sample_output_list[sample_i][sample_length - 1], output_state_j) for sample_i in range(len(sample_output_list))])
            b = np.sum([forward_matrix_list[sample_i][t_i][hide_state_i] * np.dot(backward_matrix_list[sample_i][t_i + 1], hide_state_to_hide_state[hide_state_i, :]) for t_i in range(sample_length - 1) for sample_i in range(len(sample_output_list))])
            b += np.sum([forward_matrix_list[sample_i][sample_length - 1][hide_state_i] for sample_i in range(len(sample_output_list))])
            new_hide_state_to_output_state[hide_state_i][output_state_j] = (a + epsilon) / (b + epsilon * output_state_num)

    return new_hide_state_start, new_hide_state_to_hide_state, new_hide_state_to_output_state

sample_output_list = [[0, 1, 2], ]

for _ in range(1000):
    print(_)
    hide_state_start, hide_state_to_hide_state, hide_state_to_output_state = iter_func(sample_output_list, hide_state_start, hide_state_to_hide_state, hide_state_to_output_state)
    print("_" * 100)

print("start", hide_state_start)
print("h to h", hide_state_to_hide_state)
print("h to o", hide_state_to_output_state)
~~~
