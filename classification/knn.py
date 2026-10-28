import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier

from .metrics import calculate_auc, calculate_f1, calculate_top_k_accuracy


def run_knn(X, y, metric, k_values=range(1, 16), n_splits=10):
    results = {"k": [], "auc": [], "f1_macro": [], "top_k_accuracy": []}

    labels = np.unique(y)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for k in k_values:
        print(f"Testing k = {k} ({metric} distance)...")
        auc_scores, f1_scores, top_k_scores = [], [], []

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            knn = KNeighborsClassifier(n_neighbors=k, metric=metric)
            knn.fit(X_train, y_train)

            y_pred = knn.predict(X_test)
            y_proba = knn.predict_proba(X_test)

            auc_scores.append(calculate_auc(y_test, y_proba, labels))
            f1_scores.append(calculate_f1(y_test, y_pred))
            top_k_scores.append(calculate_top_k_accuracy(y_test, y_proba, k, labels))

        results["k"].append(k)
        results["auc"].append(np.nanmean(auc_scores))
        results["f1_macro"].append(np.mean(f1_scores))
        results["top_k_accuracy"].append(np.nanmean(top_k_scores))

    return results
