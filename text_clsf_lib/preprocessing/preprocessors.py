from typing import List, Tuple

from text_clsf_lib.preprocessing.cleaning.data_cleaners import TextCleaner, DataCleaner
from text_clsf_lib.preprocessing.vectorization.data_vectorizers import DataVectorizer
from text_clsf_lib.preprocessing.vectorization.text_vectorizers import LoadedTextVectorizer


class DataPreprocessor:
    """
    This class is meant to combine data cleaning and data vectorization.
    Together it should deliver the whole process of preprocessing the data.
    fit(...) should perform fitting for data cleaner and data vectorizer.
    preprocess(...) should clean the data first, and then vectorize.

    """
    def __init__(self, data_cleaner: DataCleaner, data_vectorizer: DataVectorizer):
        self.data_cleaner = data_cleaner
        self.data_vectorizer = data_vectorizer

    def clean(self, data: List[dict]):
        return self.data_cleaner.clean(data)

    def fit(self, cleaned_data: List[dict]):
        texts, outputs = cleaned_data
        self.data_vectorizer.fit(texts, outputs)

    def save(self, save_dir):
        self.data_cleaner.save(save_dir)
        self.data_vectorizer.save(save_dir)

    def vectorize(self, data: Tuple[list, list]):
        texts, outputs = data
        return self.data_vectorizer.vectorize(texts, outputs)


class RealDataPreprocessor:
    """
    This class allows preprocessing texts for prediction from any source.
    It uses DataCleaner and LoadedTextVectorizer, which has already been trained on data to perform preprocessing.
    """
    def __init__(self, text_cleaner: TextCleaner, loaded_text_vectorizer: LoadedTextVectorizer):
        self.text_cleaner = text_cleaner
        self.text_vectorizer = loaded_text_vectorizer

    def clean(self, data: str or List[str]):
        if type(data) is str:
            data = [data]
        return self.text_cleaner.clean(texts=data)

    def vectorize(self, data: str or List[str]):
        if type(data) is str:
            data = [data]
        vectorized, cutoff_ratios = self.text_vectorizer.vectorize(texts=data)
        return vectorized, cutoff_ratios

    def clean_vectorize(self, data: str or List[str]):
        data = self.clean(data)
        return self.vectorize(data)
