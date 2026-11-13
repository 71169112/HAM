"""Compute overall and per-class metrics from three-class probabilities."""
import numpy as np
from sklearn import metrics


def evaluate_probabilities(categories, probabilities):
    categories = np.asarray(categories, dtype=np.int64)
    probabilities = np.asarray(probabilities)
    predicted = probabilities.argmax(axis=1)
    results = {'accuracy': float(metrics.accuracy_score(categories, predicted))}
    for mode in ('micro', 'macro', 'weighted'):
        results[f'{mode}_f1'] = float(metrics.f1_score(
            categories, predicted, labels=[0, 1, 2], average=mode, zero_division=0))
    precision, recall, f1, _ = metrics.precision_recall_fscore_support(
        categories, predicted, labels=[0, 1, 2], zero_division=0)
    results['macro_precision'] = float(precision.mean())
    results['macro_recall'] = float(recall.mean())
    for column, title in enumerate(('na', 'biomarker', 'therapeutic')):
        results.update({f'{title}_precision': float(precision[column]),
                        f'{title}_recall': float(recall[column]), f'{title}_f1': float(f1[column])})
    binary_labels = np.eye(3)[categories]
    results['macro_auc'] = (float(metrics.roc_auc_score(
        categories, probabilities, multi_class='ovr', average='macro'))
        if np.unique(categories).size == 3 else None)
    results['macro_aupr'] = float(np.mean([
        metrics.average_precision_score(binary_labels[:, column], probabilities[:, column])
        if binary_labels[:, column].any() else 0.0 for column in range(3)
    ]))
    return results
