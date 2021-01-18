import numpy as np


class ModelPrediction:
    def __init__(self, text: str, prediction: np.array, true_label: int):
        self.text = text
        self.prediction = prediction
        self.true_label = true_label

    def __lt__(self, other):
        return self.true_label_probability < other.true_label_probability

    @property
    def predicted_label(self):
        return np.argmax(self.prediction)

    @property
    def true_label_probability(self):
        return self.prediction[self.true_label]

    def is_correct(self):
        return self.predicted_label == self.true_label

    def __str__(self):
        return ('-----------------------------------\n'
                f'Processed text: {self.text}       \n'
                f'Prediction:     {self.prediction} \n'
                f'True label:     {self.true_label} \n'
                '-----------------------------------\n')

    def __repr__(self):
        return str(self)
