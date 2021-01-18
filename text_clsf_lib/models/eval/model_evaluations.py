from collections import defaultdict
from typing import List
import numpy as np
from sortedcontainers.sortedlist import SortedList
from sklearn import metrics
from text_clsf_lib.models.eval.model_prediction import ModelPrediction
import matplotlib
from sklearn.metrics import classification_report
from text_clsf_lib.models.eval.plots import _plot_multiple_precision_recall_curves, _plot_multiple_roc_curves, \
    _plot_multiple_conf_matrices, _plot_precision_recall, _plot_roc_curve, _plot_confusion_matrix, _plot_model_metrics



def deep_samples_test(
        texts: List[str],
        predictions: List[np.array],
        true_labels: List[int],
        show_all: bool = False,
        show_worst_samples: int or None = None,
        show_best_samples: int or None = None,
        show_wrong_samples_only: bool = False,
        to_file: None or str = None):
    """
    This test enables preview of model predictions for each sample.
    It is possible to run this function in a few different modes.
    1. show_all = True:
    Run deep_samples_test for all provided samples.
    2. show_worst_samples = n (int_value): Run deep_samples_test for n WORST model predictions.(Biggest model failures).
    3. show_best_samples = n (int_value): Run deep_samples_test for n BEST model predictions.(Biggest model success).
    4. show_wrong_samples_only = True: Run deep_samples_test for wrong samples only.
    to_file = 'path/to/your/file' instead of std output, the results will be stored to a file.

    :param texts: List[str] texts for the user preview only.
    :param predictions:
    :param true_labels:
    :param show_all:
    :param show_worst_samples:
    :param show_best_samples:
    :param show_wrong_samples_only:
    :param to_file:
    :return:
    """
    model_predictions = []
    for text, prediction, true_label in zip(texts, predictions, true_labels):
        model_predictions.append(ModelPrediction(text, prediction, true_label))

    to_file_str = f' to file: {to_file}' if to_file is not None else ''
    if show_all:
        print(f'Showing all samples{to_file_str}...')
        _show_samples(model_predictions, to_file)

    if show_wrong_samples_only:
        print(f'Showing only wrong samples{to_file_str}...')
        wrong_model_predictions = [model_pred for model_pred in model_predictions if not model_pred.is_correct()]
        _show_samples(wrong_model_predictions, to_file)

    if show_worst_samples is not None:
        print(f'Showing {show_worst_samples} worst samples{to_file_str}...')
        worst_predictions = _get_top_n_samples(model_predictions=model_predictions, n=show_worst_samples, best=False)
        _show_samples(worst_predictions, to_file)

    if show_best_samples is not None:
        print(f'Showing {show_best_samples} best samples{to_file_str}...')
        best_predictions = _get_top_n_samples(model_predictions=model_predictions, n=show_best_samples, best=True)
        _show_samples(best_predictions, to_file)


def _show_samples(sample_predictions: List[ModelPrediction], to_file):
    if to_file is not None:
        predictions_str = '\n'.join([str(prediction) for prediction in sample_predictions])
        with open(to_file, 'w') as w_file:
            w_file.write(predictions_str)
    else:
        for prediction in sample_predictions:
            print(prediction)


def _get_top_n_samples(model_predictions: List[ModelPrediction], n: int, best: bool):
    top_n_samples = SortedList(key=lambda sample: -sample.true_label_probability) if best else SortedList()
    for model_prediction in model_predictions:
        if best == model_prediction.is_correct():
            if len(top_n_samples) < n:
                top_n_samples.add(model_prediction)
            else:
                if best != (model_prediction < top_n_samples[-1]):
                    top_n_samples.pop()
                    top_n_samples.add(model_prediction)
    return [sample for sample in top_n_samples]  # so that it returns a normal list instead of SortedList


def metrics_test_multiple_models(model_output_true_label: dict,
                                 plot_precision_recall=False,
                                 plot_roc_curve=False,
                                 plot_conf_matrix=False,
                                 plot_model_metrics=False) -> dict:
    """
    This function performs metric test for many different models at the same time.
    It lets the user preview different metrics on these models and also view/save plots for these metrics.
    :param model_output_true_label: dict.
    Should look like this:
    {   'model_name1': {'predictions': <model_predictions>, 'true_labels': <true_labels>},
        'model_name2': {'predictions': <model_predictions>, 'true_labels': <true_labels>},
        ...
    }

    :param plot_precision_recall: bool. Plots precision_recall curve if set to True.
    :param plot_roc_curve: bool. Plots ROC curve if set to True.
    :param plot_conf_matrix: bool. Plots Confusion Matrices if set to True.
    :param plot_model_metrics: bool. Plots precision, recall, f1-score and roc_auc_score if set to True.
    :return: dict. Metrics and confusion matrix for models.
    """
    model_metrics = defaultdict(dict)
    model_confusion_matrices = defaultdict(dict)
    model_curves = defaultdict(dict)
    for model_name in model_output_true_label:
        true_labels = model_output_true_label[model_name]['true_labels']
        predictions = model_output_true_label[model_name]['predictions']
        offensive_predictions = [prediction[1] for prediction in predictions]
        pred_labels = [np.argmax(prediction) for prediction in predictions]
        if plot_precision_recall:
            model_curves['precision_recall'][model_name] = metrics.precision_recall_curve(true_labels,
                                                                                          offensive_predictions)
        if plot_roc_curve:
            model_curves['roc'][model_name] = metrics.roc_curve(true_labels, offensive_predictions)

        if plot_conf_matrix:
            model_curves['confusion_matrix'][model_name] = metrics.confusion_matrix(true_labels, pred_labels)

        model_metrics[model_name]['precision'] = metrics.precision_score(true_labels, pred_labels, average='macro')
        model_metrics[model_name]['recall'] = metrics.recall_score(true_labels, pred_labels, average='macro')
        model_metrics[model_name]['f1_score'] = metrics.f1_score(true_labels, pred_labels, average='macro')
        model_metrics[model_name]['roc_auc_score'] = metrics.roc_auc_score(true_labels, pred_labels, average='macro')
        model_confusion_matrices[model_name]['confusion_matrix'] = metrics.confusion_matrix(true_labels, pred_labels)

    model_metrics = dict(model_metrics)
    if plot_precision_recall:
        _plot_multiple_precision_recall_curves(model_curves['precision_recall'])

    if plot_roc_curve:
        _plot_multiple_roc_curves(model_curves['roc'])

    if plot_conf_matrix:
        _plot_multiple_conf_matrices(model_curves['confusion_matrix'])

    if plot_model_metrics:
        _plot_model_metrics(model_metrics)

    for model_name in model_metrics:
        model_metrics[model_name].update(model_confusion_matrices[model_name])
    return model_metrics


def metrics_test(predictions, true_labels,
                 plot_precision_recall=False,
                 plot_roc_curve=False,
                 plot_conf_matrix=False,
                 output_idx: int = 1) -> dict:
    """
    Metrics test for a single model.
    Provides: precision, recall, f1-score, roc_auc_score, confusion matrix.
    :param predictions: model predictions
    :param true_labels: true labels
    :param plot_precision_recall: plots precision recall curve if set to True.
    :param plot_roc_curve: plots ROC curve if set to True.
    :param plot_conf_matrix: plots confusion matrix if set to True.
    :param output_idx: for which output plots should be generated.
    :return: dict containing metrics for provided predictions and true labels.
    """
    pred_labels = [np.argmax(prediction) for prediction in predictions]
    if plot_precision_recall:
        _plot_precision_recall(predictions, true_labels, output_idx)
    if plot_roc_curve:
        _plot_roc_curve(predictions, true_labels, output_idx)
    if plot_conf_matrix:
        _plot_confusion_matrix(pred_labels, true_labels)
    return {
        'precision': metrics.precision_score(true_labels, pred_labels, average='macro'),
        'recall': metrics.recall_score(true_labels, pred_labels, average='macro'),
        'f1-score': metrics.f1_score(true_labels, pred_labels, average='macro'),
        'roc_auc_score': metrics.roc_auc_score(true_labels, pred_labels, average='macro'),
        'confusion_matrix': metrics.confusion_matrix(true_labels, pred_labels)
    }