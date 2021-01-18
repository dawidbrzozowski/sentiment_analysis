from math import sqrt

import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

"""
This module provides plotting options for metric tests.
These functions are meant to be used from inside metric tests only.
It is not recommended to use them in other cases.
"""


def _plot_model_metrics(model_metrics: dict):
    metric_names = [metric_name for metric_name in next(iter(model_metrics.values()))]
    x = np.arange(len(metric_names))
    width_per_score = 0.76

    fig, ax = plt.subplots()
    fig.canvas.set_window_title('Model scores')

    rects = []
    models_amount = len(model_metrics)
    width_per_model = width_per_score / models_amount
    for i, model_name in enumerate(model_metrics):
        bias = i * width_per_model
        model_scores = [round(value, 2) for value in model_metrics[model_name].values()]
        rects.append(ax.bar(x - width_per_model * models_amount / 2 + bias, model_scores, width_per_model,
                            label=model_name, align='edge'))

    ax.set_ylabel('Scores')
    ax.set_title('Model scores (macro)')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    def label_rects(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate(str(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 2),  # 3 points vertical offset
                        fontsize=8,
                        textcoords="offset points",
                        ha='center', va='bottom')

    for rect in rects:
        label_rects(rect)

    fig.tight_layout()
    plt.show()


def _plot_multiple_precision_recall_curves(precision_recall_curves: dict):
    curves_amount = len(precision_recall_curves)
    fig, axs, rows_amount, columns_amount = _setup_plot(curves_amount=curves_amount,
                                                        plot_title='Precision Recall Curves')
    i = None
    for i, model_name in enumerate(precision_recall_curves):
        precisions, recalls, thresholds = precision_recall_curves[model_name]
        axs[i // columns_amount][i % columns_amount].set_title(model_name)
        axs[i // columns_amount][i % columns_amount].plot(thresholds, precisions[:-1], 'b--', label='Precision')
        axs[i // columns_amount][i % columns_amount].plot(thresholds, recalls[:-1], 'g-', label='Recall')
        axs[i // columns_amount][i % columns_amount].set_xlabel('Threshold')
    if i is not None:
        handles, labels = axs[i // columns_amount][i % columns_amount].get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower right')
    _switch_off_unused_subplots(axs, curves_amount, rows_amount, columns_amount)

    plt.tight_layout()
    plt.ylim([0, 1])
    plt.show()


def _plot_multiple_roc_curves(roc_curves: dict):
    curves_amount = len(roc_curves)
    fig, axs, rows_amount, columns_amount = _setup_plot(curves_amount=curves_amount,
                                                        plot_title='ROC Curves')

    for i, model_name in enumerate(roc_curves):
        fpr, tpr, thresholds = roc_curves[model_name]
        axs[i // columns_amount][i % columns_amount].set_title(model_name)
        axs[i // columns_amount][i % columns_amount].plot(fpr, tpr, linewidth=2)
        axs[i // columns_amount][i % columns_amount].plot([0, 1], [0, 1], 'k--')
        axs[i // columns_amount][i % columns_amount].axis([0, 1, 0, 1])
        axs[i // columns_amount][i % columns_amount].set_xlabel('False Positive Rate')
        axs[i // columns_amount][i % columns_amount].set_ylabel('True Positive Rate')
    _switch_off_unused_subplots(axs, curves_amount, rows_amount, columns_amount)

    plt.tight_layout()
    plt.show()


def _plot_multiple_conf_matrices(confusion_matrices: dict):
    curves_amount = len(confusion_matrices)
    fig, axs, rows_amount, columns_amount = _setup_plot(curves_amount=curves_amount,
                                                        plot_title='Confusion Matrices')
    for i, model_name in enumerate(confusion_matrices):
        confusion_matrix = confusion_matrices[model_name]
        row_sums = confusion_matrix.sum(axis=1, keepdims=True)
        normalized_confusion_matrix = confusion_matrix / row_sums
        np.fill_diagonal(normalized_confusion_matrix, 0)
        axs[i // columns_amount][i % columns_amount].matshow(normalized_confusion_matrix, cmap=plt.cm.get_cmap('gray'))
        axs[i // columns_amount][i % columns_amount].set_title(model_name)
    _switch_off_unused_subplots(axs, curves_amount, rows_amount, columns_amount)

    plt.tight_layout()
    plt.show()


def _setup_plot(curves_amount, plot_title):
    rows_amount = round(sqrt(curves_amount))
    columns_amount = rows_amount + 1 if rows_amount * rows_amount < curves_amount else rows_amount
    fig, axs = plt.subplots(rows_amount, columns_amount, squeeze=False)
    fig.canvas.set_window_title(plot_title)
    fig.suptitle(plot_title)
    return fig, axs, rows_amount, columns_amount


def _switch_off_unused_subplots(axs, curves_amount, rows_amount, columns_amount):
    for i in range(curves_amount, rows_amount * columns_amount):
        axs[i // rows_amount, i % columns_amount].axis('off')


def _plot_precision_recall(predictions, true_labels, idx):
    scores = [prediction[idx] for prediction in predictions]
    precisions, recalls, thresholds = metrics.precision_recall_curve(true_labels, scores)
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Threshold')
    plt.legend(loc='center right')
    plt.title('Precision Recall Curve')
    fig = plt.gcf()
    fig.canvas.set_window_title('Precision Recall Curve')
    plt.ylim([0, 1])
    plt.show()


def _plot_roc_curve(predictions, true_labels, idx):
    scores = [prediction[idx] for prediction in predictions]
    fpr, tpr, thresholds = metrics.roc_curve(true_labels, scores)
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    fig = plt.gcf()
    fig.canvas.set_window_title('ROC Curve')
    plt.show()


def _plot_confusion_matrix(pred_labels, true_labels):
    confusion_matrix = metrics.confusion_matrix(true_labels, pred_labels)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion_matrix / row_sums
    np.fill_diagonal(normalized_confusion_matrix, 0)
    plt.matshow(normalized_confusion_matrix, cmap=plt.cm.get_cmap('gray'))
    plt.title('Confusion Matrix')
    fig = plt.gcf()
    fig.canvas.set_window_title('Confusion Matrix')
    plt.show()


if __name__ == '__main__':
    model_metrics = {
        'glove_rnn': {
            'precision': np.float(0.567653),
            'recall': 0.5,
            'roc': 0.4,
            'f1': 0.5,
        },
        'tfidf': {
            'precision': 0.32,
            'recall': 0.5,
            'roc': 0.3,
            'f1': 0.8

        },
        'sds': {
            'precision': 0.32,
            'recall': 0.5,
            'roc': 0.3,
            'f1': 0.8

        }
    }
    _plot_model_metrics(model_metrics)
