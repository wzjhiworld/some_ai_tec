# CTC 概率值的计算

## 代码示例

~~~python
import numpy as np
from sklearn import preprocessing

# 输出的概率的每一位对应的字符，第一位表示空
char_list = ['-', 'a', 'b']

# prob_matrix = preprocessing.normalize(np.random.random((5, 3)), "l1")
prob_matrix = np.array([[0.1, 0.8, 0.1],
                        [0.2, 0.7, 0.1],
                        [0.1, 0.2, 0.7]], dtype="float32")

target_list = 'ab'
target_list = "".join(['-' + e for e in target_list]) + '-'
target_list = [e for e in target_list]
print(target_list)

def get_prob(prob_matrix, target_list, char_list:list):
    input_length = prob_matrix.shape[0]
    output_length = len(target_list)
    out_prob = np.zeros((input_length, output_length))
    out_prob[0][0] = prob_matrix[0][char_list.index('-')]
    out_prob[0][1] = prob_matrix[0][char_list.index(target_list[1])]
    for i in range(1, input_length):
        for j in range(0, output_length):
            t_char = target_list[j]
            t_char_index = char_list.index(t_char)
            if t_char == '-':
                out_prob[i][j] += out_prob[i - 1][j] * prob_matrix[i][t_char_index]
                out_prob[i][j] += out_prob[i - 1][j - 1] * prob_matrix[i][t_char_index]
            else:
                out_prob[i][j] += out_prob[i - 1][j] * prob_matrix[i][t_char_index]
                out_prob[i][j] += out_prob[i - 1][j - 1] * prob_matrix[i][t_char_index]
                if j >= 2 and target_list[j - 2] != t_char:
                    out_prob[i][j] += out_prob[i - 1][j - 2] * prob_matrix[i][t_char_index]
    return out_prob[-1][-1] + out_prob[-1][-2]

print(get_prob(prob_matrix, target_list, char_list))
~~~
