from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix
import numpy as np
from scipy.stats import mode

def gmm_fault_separation(train_data, train_label,test_data, test_label, n_classes):
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
        true_labels = test_label[cluster_indexes]
        most_common_label = mode(true_labels)[0][0]
        accuracy = np.sum(true_labels == most_common_label) / len(true_labels)
        category_accuracy.append(accuracy)
    return category_accuracy



