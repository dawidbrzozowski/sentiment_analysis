from abc import abstractmethod
from typing import List

import os

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from text_clsf_lib.preprocessing.vectorization.embeddings.embedding_loaders import WordEmbeddingsLoader
from text_clsf_lib.preprocessing.vectorization.embeddings.matrix_preparer import EmbeddingsMatrixPreparer
from text_clsf_lib.preprocessing.vectorization.embeddings.text_encoders import TextEncoder, LoadedTextEncoder
from text_clsf_lib.utils.files_io import write_pickle, write_numpy, read_pickle, load_json, append_json
import numpy as np
from bpemb import BPEmb

VECTORIZER_NAME = 'vectorizer.vec'
EMBEDDING_MATRIX_NAME = 'embedding_matrix.npy'


class TextVectorizer:
    """
    Base class for TextVectorizers.
    """

    @abstractmethod
    def fit(self, texts: List[str]):
        pass

    @abstractmethod
    def vectorize(self, texts: List[str]):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass


class BagOfWordsTextVectorizer(TextVectorizer):
    def __init__(self, max_features):
        self.bag_of_words_vec = CountVectorizer(max_features=max_features)

    def fit(self, texts: List[str]):
        self.bag_of_words_vec.fit(texts)

    def save(self, save_dir):
        os.makedirs(f'{save_dir}', exist_ok=True)
        write_pickle(f'{save_dir}/{VECTORIZER_NAME}', self.bag_of_words_vec)
        sorted_vocab = [item[0] for item in sorted(self.bag_of_words_vec.vocabulary_.items(), key=lambda item: item[1])]
        with open(f'{save_dir}/vocab.txt', 'w') as f:
            for word in sorted_vocab:
                f.write(f'{word}\n')
        predictor_config = {'vectorizer': {'type': 'count'}}
        append_json(f'{save_dir}/predictor_config.json', predictor_config)

    def vectorize(self, texts: List[str]):
        return self.bag_of_words_vec.transform(texts).toarray()


class TfIdfTextVectorizer(TextVectorizer):
    def __init__(self, max_features):
        self.tfidf_vec = TfidfVectorizer(max_features=max_features)

    def fit(self, texts: List[str]):
        self.tfidf_vec.fit(texts)

    def save(self, save_dir):
        os.makedirs(f'{save_dir}', exist_ok=True)
        write_pickle(f'{save_dir}/{VECTORIZER_NAME}', self.tfidf_vec)
        sorted_vocab = [item[0] for item in sorted(self.tfidf_vec.vocabulary_.items(), key=lambda item: item[1])]
        np.savetxt(f'{save_dir}/idf.txt', self.tfidf_vec.idf_, fmt='%f')
        with open(f'{save_dir}/vocab.txt', 'w') as f:
            for word in sorted_vocab:
                f.write(f'{word}\n')
        predictor_config = {'vectorizer': {'type': 'count'}}
        append_json(f'{save_dir}/predictor_config.json', predictor_config)

    def vectorize(self, texts: List[str]):
        return self.tfidf_vec.transform(texts).toarray()


class GloveEmbeddingTextVectorizer(TextVectorizer):
    def __init__(self, max_vocab_size, max_seq_len, embedding_dim, embedding_type):
        self.text_encoder = TextEncoder(max_vocab_size=max_vocab_size, max_seq_len=max_seq_len)
        self.embedding_dim = embedding_dim
        self.embedding_matrix = None
        self.embeddings_loader = WordEmbeddingsLoader(embedding_type)

    def fit(self, texts: List[str]):
        self.text_encoder.fit(texts)
        word2vec = self.embeddings_loader.load_word_vectors(self.embedding_dim)
        emb_matrix_preparer = EmbeddingsMatrixPreparer(self.text_encoder.word2idx, word2vec)
        self.embedding_matrix = emb_matrix_preparer.prepare_embedding_matrix()

    def save(self, save_dir):
        os.makedirs(f'{save_dir}', exist_ok=True)
        write_numpy(f'{save_dir}/{EMBEDDING_MATRIX_NAME}', self.embedding_matrix)
        self.text_encoder.save(save_dir)

    def vectorize(self, texts: List[str]):
        return self.text_encoder.encode(texts)


class BPEEmbeddingTextVectorizer(TextVectorizer):
    def __init__(self, embedding_dim, max_seq_len, max_vocab_size, **kwargs):
        self.bpemb = BPEmb(lang='en', vs=max_vocab_size, dim=embedding_dim, add_pad_emb=True)
        self.max_seq_len = max_seq_len

    def fit(self, texts: List[str]):
        pass

    def save(self, save_dir):
        os.makedirs(f'{save_dir}', exist_ok=True)
        write_numpy(f'{save_dir}/{EMBEDDING_MATRIX_NAME}', self.bpemb.vectors)
        predictor_config = {
            'vectorizer': {
                'max_seq_len': self.max_seq_len,
                'embedding_dim': self.bpemb.dim,
                'max_vocab_size': self.bpemb.vs,
                'type': 'bpe_embedding'
            }
        }
        append_json(f'{save_dir}/predictor_config.json', predictor_config)
        with open(f'{save_dir}/vocab.txt', 'w') as f:
            for word in self.bpemb.words:
                f.write(f'{word}\n')

    def vectorize(self, texts: List[str]):
        return _vectorize_padded(bpemb=self.bpemb, max_seq_len=self.max_seq_len, texts=texts)


def _vectorize_padded(bpemb, max_seq_len, texts: List[str]):
    ids_not_padded = bpemb.encode_ids(texts)
    ids_padded = []
    max_len_found = 0
    for text_enc in ids_not_padded:
        if len(text_enc) > max_len_found: max_len_found = len(text_enc)
        if len(text_enc) > max_seq_len:
            text_enc = text_enc[:max_seq_len]
        elif len(text_enc) < max_seq_len:
            padding = [len(bpemb.words) - 1] * (max_seq_len - len(text_enc))
            text_enc.extend(padding)
        ids_padded.append(text_enc)
    padded = np.array(ids_padded)
    print(f'Max len: {max_len_found}')
    return padded


class LoadedTextVectorizer:
    """
    This is a base class for LoadedTextVectorizers.
    Their goal is to vectorize data, using preprocessing files created during training.
    """

    @abstractmethod
    def vectorize(self, texts: List[str]):
        pass


class LoadedEmbeddingTextVectorizer(LoadedTextVectorizer):
    def __init__(self, predictor_config, tokenizer):
        self.text_encoder = LoadedTextEncoder(tokenizer=tokenizer,
                                              predictor_config=predictor_config)

    def vectorize(self, texts: List[str]):
        vectorized, cut_off_ratios = self.text_encoder.encode(texts)
        return vectorized, cut_off_ratios


class LoadedTfIdfTextVectorizer(LoadedTextVectorizer):
    def __init__(self, vectorizer):
        self.tfidf_vec = vectorizer

    def vectorize(self, texts: List[str]):
        return self.tfidf_vec.transform(texts).toarray(), [1 for _ in range(len(texts))]


class LoadedBPEEmbeddingTextVectorizer(LoadedTextVectorizer):
    def __init__(self, predictor_config):
        predictor_config = predictor_config['vectorizer']
        self.bpemb = BPEmb(lang='en', dim=predictor_config['embedding_dim'],
                           vs=predictor_config['max_vocab_size'], add_pad_emb=True)
        self.max_seq_len = predictor_config['max_seq_len']

    def get_cutoff_ratios(self, texts: List[str]) -> List[float]:
        sequences = self.bpemb.encode_ids(texts)
        return [len(sequence) / self.max_seq_len for sequence in sequences]

    def vectorize(self, texts: List[str]):
        vectorized = _vectorize_padded(bpemb=self.bpemb, max_seq_len=self.max_seq_len, texts=texts)
        cut_off_ratios = self.get_cutoff_ratios(texts)
        return vectorized, cut_off_ratios
