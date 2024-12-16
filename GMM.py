from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import numpy as np


def gmm_fault_separation(data, labels, n_classes):
    """
    Using GMM to calculate the isolation score of each fault
    :param data: fault data
    :param labels: fault label
    :param n_classes: fault class
    :param threshold: the threshold that faults is considered almost isolated （default value:98%）
    :return: fault almost isolated
    """
    # train GMM model
    gmm = GaussianMixture(n_components=n_classes, random_state=0)
    gmm_labels = gmm.fit_predict(data.reshape(-1, 1))
    # calculated confusion matrix
    conf_matrix = confusion_matrix(labels, gmm_labels, labels=np.arange(n_classes))
    # calculated the accuracy of each fault
    separation_accuracy = conf_matrix.max(axis=1) / conf_matrix.sum(axis=1)

    return separation_accuracy


# 示例数据
np.random.seed(42)
n_samples = 1000  # 样本数
n_classes = 13  # 包括12类故障和1类正常信号
n_measurements = 4  # 测点数

# 模拟数据：每个故障类别生成不同均值和标准差的数据
data = []
labels = []
for i in range(n_classes):
    class_data = np.random.normal(loc=i * 2, scale=1.0, size=n_samples // n_classes)
    data.append(class_data)
    labels += [i] * (n_samples // n_classes)

data = np.concatenate(data)
labels = np.array(labels)

# 测点模拟：每列表示一个测点
measurements = np.random.normal(loc=data[:, None], scale=0.5, size=(len(data), n_measurements))

# 选择第一个测点进行 GMM 分离能力分析
selected_measurement = measurements[:, 0]
separated_classes, count = gmm_fault_separation(selected_measurement, labels, n_classes)

print("能够分离的故障类别:", separated_classes)
print("能够分离的故障类别数量:", count)
