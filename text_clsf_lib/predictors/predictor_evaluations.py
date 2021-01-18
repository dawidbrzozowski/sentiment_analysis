from typing import List

from text_clsf_lib.predictors.predictor import Predictor
import numpy as np
from lime.lime_text import LimeTextExplainer

from text_clsf_lib.predictors.presets import create_predictor_preset


def deep_predictor_test_on_sample_own_impl(predictor: Predictor, text: str, desired_label=1) -> List[tuple]:
    """
    This function should be used for analyzing predictions made by predictor.
    The function will print the importance of each word when making a prediction.
    :param predictor: Predictor object.
    :param text: Text to be analyzed.
    :param desired_label: Label that you are interested in.
     For example, if you want to analyze why is the text offensive and model output is [NotOffensive, Offensive],
     then the desired_label should be equal to 1.
    :return: Words from the text sorted by their impact on the prediction.
    In case of offensive language classification, this would be words from the most offensive to the least one.
    """
    print(f'Probabilities on text: {text}')
    whole_sentence_prob = predictor.predict(text)[0][desired_label]
    words = text.split()
    print(whole_sentence_prob)
    probs_without_word = {}
    for word in words:
        subtext = text.replace(word, '')
        print(f'Probabilities on text: {subtext}. \n Word ommited: {word}')
        probs_without_word[word] = predictor.predict(subtext)[0][desired_label]
        print(probs_without_word[word])
    std_deviation = np.std([probs_without_word[word] for word in words])
    impact_on_text = {word: (probs_without_word[word] - whole_sentence_prob) / std_deviation for word in words}
    return sorted(impact_on_text.items(), key=lambda item: item[1])


class LimePredictor:
    def __init__(self, predictor: Predictor):
        self.preprocessor = predictor.preprocessor
        self.model_runner = predictor.model_runner

    def predict(self, text: list or str):
        preprocessed = self.preprocessor.clean_vectorize(text)[0]
        return self.model_runner.run(preprocessed)


def deep_test_on_sample_lime(predictor: Predictor,
                             text: str,
                             labels_to_explain: List[int],
                             true_label=None):
    lime_predictor = LimePredictor(predictor)
    explainer = LimeTextExplainer()
    explained_instance = explainer.explain_instance(
        text_instance=text,
        classifier_fn=lime_predictor.predict,
        labels=labels_to_explain)
    print(f'Explaining text: {text}')
    print(f'Model prediction: {lime_predictor.predict(text)}')
    print(f'True label: {true_label}')
    for label in labels_to_explain:
        print(f'Explanation for label: {label}')
        print('\n'.join(map(str, explained_instance.as_list(label=label))))


if __name__ == '__main__':
    text = "You look like a monkey!"
    preset = create_predictor_preset('bpe_best_model')
    predictor = Predictor(preset)
    deep_test_on_sample_lime(predictor=predictor,
                             text=text,
                             labels_to_explain=[1],
                             true_label=0)
