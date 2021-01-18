from text_clsf_lib.preprocessing.vectorization.output_vectorizers import OutputVectorizer
from text_clsf_lib.preprocessing.vectorization.text_vectorizers import TextVectorizer


class DataVectorizer:
    """
    This class is meant to vectorize X and y (texts and outputs).
    To perform that, it uses TextVectorizer and OutputVectorizer.
    vectorize(...) method should return X and y vectorized.
    """
    def __init__(self, text_vectorizer: TextVectorizer, output_vectorizer: OutputVectorizer):
        self.text_vectorizer = text_vectorizer
        self.output_vectorizer = output_vectorizer

    def fit(self, texts, outputs):
        self.text_vectorizer.fit(texts)
        self.output_vectorizer.fit(outputs)

    def vectorize(self, texts, outputs):
        return self.text_vectorizer.vectorize(texts), self.output_vectorizer.vectorize(outputs)

    def save(self, save_dir):
        self.text_vectorizer.save(save_dir)
