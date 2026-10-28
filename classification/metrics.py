import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, top_k_accuracy_score


def calculate_auc(y_test, y_proba, labels):
    if len(np.unique(y_test)) <= 1:
        return np.nan
    return roc_auc_score(y_test, y_proba, multi_class="ovr", labels=labels)


def calculate_f1(y_test, y_pred):
    return f1_score(y_test, y_pred, average="macro", zero_division=0)


def calculate_top_k_accuracy(y_test, y_proba, k, labels):
    """Top-k accuracy, com k limitado a (n_classes - 1)."""
    n_classes = len(labels)
    if n_classes <= 1:
        return np.nan

    k_metric = max(1, min(k, n_classes - 1))
    return top_k_accuracy_score(y_test, y_proba, k=k_metric, labels=labels)
