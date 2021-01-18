from abc import abstractmethod
from typing import List

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
from text_clsf_lib.utils.files_io import write_pickle, read_pickle, load_json, append_json
import os

TOKENIZER_NAME = 'tokenizer.pickle'


class TextEncoderBase:
    """
    This class is a base of Text Encoders.
    It's goal is to convert text into integers.
    fit(...) method should be used to determine most frequent words, that should appear in word2idx.
    encode(...) method should convert texts into ndarray of shape (N_TEXTS x MAX_SEQ_LEN)
    """

    def __init__(self, max_vocab_size, max_seq_len):
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len

    @abstractmethod
    def fit(self, texts: List[str]):
        pass

    @abstractmethod
    def encode(self, texts: List[str]):
        pass

    @abstractmethod
    def save(self, save_dir):
        pass


class TextEncoder(TextEncoderBase):
    def __init__(self, max_vocab_size, max_seq_len):
        super().__init__(max_vocab_size, max_seq_len)
        self.word2idx = None
        self.tokenizer = Tokenizer(num_words=max_vocab_size, lower=True)

    def fit(self, texts):
        self.tokenizer.fit_on_texts(texts)
        self.word2idx = {word: idx for word, idx in self.tokenizer.word_index.items() if idx < self.tokenizer.num_words}

    def save(self, save_dir):
        os.makedirs(f'{save_dir}', exist_ok=True)
        write_pickle(f'{save_dir}/{TOKENIZER_NAME}', self.tokenizer)
        vocab = [item[0] for item in self.word2idx.items()]
        with open(f'{save_dir}/vocab.txt', 'w') as f:
            f.write('<pad>\n')
            for word in vocab:
                f.write(f'{word}\n')
        predictor_config = {
            'vectorizer': {'max_seq_len': self.max_seq_len,
                           'type': 'word_embedding'}
        }
        append_json(f'{save_dir}/predictor_config.json', predictor_config)

    def encode(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences=sequences, maxlen=self.max_seq_len, padding='post', truncating='post')
        return padded


class LoadedTextEncoder:
    def __init__(self, tokenizer, predictor_config):
        self.tokenizer = tokenizer
        predictor_config = predictor_config
        self.max_seq_len = predictor_config['vectorizer']['max_seq_len']

    def encode(self, texts):
        sequences = self.tokenizer.texts_to_sequences(texts)
        padded = pad_sequences(sequences=sequences, maxlen=self.max_seq_len, padding='post', truncating='post')
        cut_off_ratios = [len(sequence)/self.max_seq_len for sequence in sequences]
        return padded, cut_off_ratios
