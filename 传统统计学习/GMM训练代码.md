# GMM 高斯混合模型代码实现

## code 代码实现

~~~python
import numpy as np

gmm_center_num = 2
gmm_dim_size = 2
epsilon = 1e-6
def get_train_samples(size = 1000):
    data = np.zeros((0, 2), dtype="float32")
    for _ in range(size):
        if np.random.random() <= 0.3:
            mean1 = [0, 0]
            cov1 = [[1, 0], [0, 10]]
            t_data = np.random.multivariate_normal(mean1, cov1, 1)
            data = np.append(data, t_data, axis = 0)
        else:
            mean1 = [10, 10]
            cov1 = [[10, 0], [0, 1]]
            t_data = np.random.multivariate_normal(mean1, cov1, 1)
            data = np.append(data, t_data, axis = 0)
    data = np.round(data, 6) #保留多少位精度
    print("generate size", data.shape)
    return data

train_data = get_train_samples()

def get_init_gmm_params(train_data):
    from sklearn.cluster import KMeans
    kmeans = KMeans(n_clusters=gmm_center_num, max_iter=100, tol=1e-4)
    kmeans.fit(train_data)
    centers = kmeans.cluster_centers_
    print(centers)
    predict_center_ids = kmeans.predict(train_data)
    ans_list = []
    for i in range(gmm_center_num):
        t_center = centers[i:i+1, :]
        t_data = train_data[predict_center_ids == i] - t_center
        t_cov = np.cov(t_data.T)
        t_weight = len(t_data) / len(train_data)
        ans_list.append((t_weight, t_center[0], t_cov, np.linalg.inv(t_cov)))
    print("init params", ans_list)
    return ans_list

init_gmm_params = get_init_gmm_params(train_data)

#随意初始化一个
init_gmm_params = [(0.2, np.array([10, 20]), np.array([[1, 0],[0, 1]]), np.array([[1, 0],[0, 1]]),), #按顺序分别为高斯分量权重， 高斯均值， 高斯协方差矩阵， 高斯协方差矩阵的逆矩阵
                   (0.8, np.array([0, 20]), np.array([[1, 0],[0, 1]]), np.array([[1, 0],[0, 1]]),),]

def gm_p_weight_calcu(gassi_params, x_vector):
    """
    这里计算 gassi 分布的概率时，没有计算前面的系数，因为这对后续计算条件概率没有影响。
    :param gassi_params:
    :param x_vector:
    :return:
    """
    center = gassi_params[1]
    cov_reverse = gassi_params[3]
    x_vector = x_vector - center
    return -0.5 * np.dot(np.dot(x_vector.reshape(1, -1), cov_reverse), x_vector.reshape(-1, 1))

def e_step(train_data, gmm_params):
    data_size = train_data.shape[0]
    p_kcenter_by_sample = np.zeros((data_size, gmm_center_num))
    center_weight_list = np.array([gmm_params[i][0] for i in range(gmm_center_num)])
    for i in range(data_size):
        t_p_weight_list = np.zeros((gmm_center_num,))
        for center_i in range(gmm_center_num):
            t_p_weight_list[center_i] = gm_p_weight_calcu(gmm_params[center_i], train_data[i])
        t_p_weight_list = t_p_weight_list - np.max(t_p_weight_list)
        t_p_weight_list = np.exp(t_p_weight_list) * center_weight_list
        p_kcenter_by_sample[i] = t_p_weight_list / np.sum(t_p_weight_list)
    return p_kcenter_by_sample

def m_step(train_data, p_kcenter_by_sample, gmm_params):
    new_ans_list = []
    for center_i in range(gmm_center_num):
        old_center = gmm_params[center_i][1]
        new_center_i = np.sum(train_data * (p_kcenter_by_sample[:, center_i:center_i+1] + epsilon / len(train_data)), axis = 0) / (np.sum(p_kcenter_by_sample[:, center_i]) + epsilon)
        tmp_t = train_data - old_center.reshape(1, -1)
        tmp_a = tmp_t.reshape(-1, gmm_dim_size, 1)
        tmp_b = tmp_t.reshape(-1, gmm_dim_size, 1)
        tmp_t = np.einsum("ijk, ilk -> ijl", tmp_a, tmp_b)
        new_cov_i = np.sum(tmp_t * (p_kcenter_by_sample[:, center_i] + epsilon / len(train_data)).reshape(-1, 1, 1), axis=0) / (np.sum(p_kcenter_by_sample[:, center_i]) + epsilon)
        new_weight_i = np.sum(p_kcenter_by_sample[:, center_i]) / len(train_data)
        new_ans_list.append((new_weight_i, new_center_i, new_cov_i, np.linalg.inv(new_cov_i)))
    return new_ans_list

iter_gmm_params = init_gmm_params
for _ in range(5000):
    print("iter ", _)
    p_kcenter_by_sample = e_step(train_data, iter_gmm_params)
    iter_gmm_params = m_step(train_data, p_kcenter_by_sample, iter_gmm_params)

print(iter_gmm_params)
~~~
