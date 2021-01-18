from collections import defaultdict
from typing import List

from sklearn.metrics import classification_report

from text_clsf_lib.models.eval.model_evaluations import deep_samples_test, metrics_test, metrics_test_multiple_models
from text_clsf_lib.models.model_data import prepare_model_data
from text_clsf_lib.models.model_trainer_runner import NNModelRunner
import numpy as np

DEEP_SAMPLES_DEFAULTS = {
    'show_worst_samples': 10,
}

METRICS_TEST_DEFAULTS = {
    'plot_precision_recall': True,
    'plot_roc_curve': True,
    'plot_conf_matrix': True
}


def test_single_model(
        preset: dict,
        plot_precision_recall: bool = True,
        plot_roc_curve: bool = True,
        plot_conf_matrix: bool = True,
        verbose: int = 1):
    """
    Wrapper function for testing model.
    Enables easy and fast model testing using the preset from training process.
    :param verbose: if set to 0, nothing will show up
    :param plot_conf_matrix: Decide whether to plot confusion matrix.
    :param plot_precision_recall: Decide whether to plot Precision Recall relations.
    :param plot_roc_curve: Decide whether to plot ROC curve.
    :param preset: preset used for model training.
    :return: Predictions of the model and true labels as tuple.
    """
    model_runner = NNModelRunner(model_path=f"{preset['model_save_dir']}/{preset['model_name']}.h5")
    data = prepare_model_data(
        data_params=preset['data_params'],
        vectorizer_params=preset['vectorizer_params'])

    data_test_vec = data['test_vectorized']
    predictions, labels = model_runner.test(data_test_vec)

    if verbose == 1:
        metrics = metrics_test(
                    predictions=predictions,
                    true_labels=labels,
                    plot_precision_recall=plot_precision_recall,
                    plot_conf_matrix=plot_conf_matrix,
                    plot_roc_curve=plot_roc_curve)
        report = classification_report(y_true=labels, y_pred=[np.argmax(prediction) for prediction in predictions])
        show_single_model_metrics(metrics=metrics, model_name=preset['model_name'], clsf_report=report)

    return predictions, labels


def show_single_model_metrics(metrics: dict, model_name: str, clsf_report):
    print(f"Showing metrics for model: {model_name}")
    print('==========================================================')
    print('Scikit-Learn Classification Report on provided test corpus')
    print('==========================================================')
    print(clsf_report)


def test_single_model_sample_analysis(
        preset: dict,
        show_all_samples: bool = False,
        show_worst_samples: int or None = None,
        show_best_samples: int or None = None,
        show_wrong_samples_only: bool = True,
        to_file: None or str = None):
    model_runner = NNModelRunner(model_path=f"{preset['model_save_dir']}/{preset['model_name']}.h5")
    data = prepare_model_data(
        data_params=preset['data_params'],
        vectorizer_params=preset['vectorizer_params'])

    data_test_vec = data['test_vectorized']
    cleaned_texts, _ = data['test_cleaned']
    predictions, labels = model_runner.test(data_test_vec)
    deep_samples_test(texts=cleaned_texts,
                      predictions=predictions,
                      true_labels=labels,
                      show_all=show_all_samples,
                      show_worst_samples=show_worst_samples,
                      show_best_samples=show_best_samples,
                      show_wrong_samples_only=show_wrong_samples_only,
                      to_file=to_file)


def test_multiple_models(presets: List[dict],
                         plot_precision_recall=True,
                         plot_roc_curve=True,
                         plot_conf_matrix=True,
                         plot_model_metrics=True):
    """
    Wrapper function for testing many models at the same time and comparing them.
    Takes list of presets, used during training process.
    Enables showing model comparison plots.
    :param presets: list of preset used during training process for the tested models.
    :param plot_precision_recall: bool
    :param plot_roc_curve: bool
    :param plot_conf_matrix: bool
    :param plot_model_metrics: bool
    :return: dict. Metrics for all models. Metrics include: precision, recall, f1-score, roc_auc_score, confusion_matrix
    """
    model_name_to_scores = defaultdict(dict)
    for preset in presets:
        predictions, labels = test_single_model(preset, verbose=0)
        model_name_to_scores[preset['model_name']]['predictions'] = predictions
        model_name_to_scores[preset['model_name']]['true_labels'] = labels
    return metrics_test_multiple_models(model_name_to_scores,
                                        plot_precision_recall=plot_precision_recall,
                                        plot_roc_curve=plot_roc_curve,
                                        plot_conf_matrix=plot_conf_matrix,
                                        plot_model_metrics=plot_model_metrics)
