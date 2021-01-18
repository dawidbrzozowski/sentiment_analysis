from abc import abstractmethod
from typing import List
import numpy as np


class OutputVectorizer:
    @abstractmethod
    def fit(self, output: List[dict]):
        pass

    @abstractmethod
    def vectorize(self, output: List[dict]):
        pass


class BasicOutputVectorizer(OutputVectorizer):
    def fit(self, output: List[dict]):
        pass

    def vectorize(self, output: List[int]):
        return np.array(output)
