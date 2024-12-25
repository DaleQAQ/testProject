from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.stats import mode
import os
import scipy.io as io
from sklearn.preprocessing import StandardScaler

def gmm_fault_separation(train_data, test_data, test_label, n_classes):
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
    gmm.fit(train_data)
    y_pred = gmm.predict(test_data)
    category_accuracy = []
    for cluster in range(n_classes):
        cluster_indexes = np.where(y_pred == cluster)[0]
        if len(cluster_indexes) > 0:
            true_labels = [test_label[cluster_index] for cluster_index in cluster_indexes]
            most_common_label = mode(true_labels)[0][0]
            accuracy = np.sum(true_labels == most_common_label) / len(true_labels)
        else:
            accuracy = 0
        category_accuracy.append(accuracy)
    return category_accuracy

def preprocessing(file):
    data_file = io.loadmat(file)
    train_data = data_file["train_data"]
    test_data = data_file["test_data"]
    test_label = data_file["test_y"]
    test_label = [np.argmax(test_y) for test_y in test_label]

    standard_scaler = StandardScaler()
    train_data = standard_scaler.fit_transform(train_data)
    test_data = standard_scaler.fit_transform(test_data)
    return train_data, test_data, test_label

def node_accuracy_dic_generate(path):
    """
    :param path: your top mat file path
    :return: node accuracy dictionary
    """
    accuracy_dic = {}
    files = os.listdir(path)
    for file in files:
        sub_path = os.path.join(path, file)
        if os.path.isfile(sub_path):
            train_data, test_data, test_label = preprocessing(sub_path)
            accuracy_this_node = gmm_fault_separation(train_data, test_data, test_label, n_classes=13)
            accuracy_dic[file[:-4]] = accuracy_this_node
    return accuracy_dic


