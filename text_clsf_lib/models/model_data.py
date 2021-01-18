from text_clsf_lib.data_preparation.data_balancing import undersample_to_even, cut_off_longer_texts
from text_clsf_lib.data_preparation.data_extracton import load_data
from text_clsf_lib.preprocessing.cleaning.data_cleaners import PresetDataCleaner, TextCleaner, OutputCleaner
from text_clsf_lib.preprocessing.preprocessors import DataPreprocessor
import os

from text_clsf_lib.utils.files_io import read_pickle, write_pickle

CACHE_FILE_NAME = 'cache_data.pickle'


def prepare_model_data(data_params: dict, vectorizer_params: dict) -> dict:
    """
    Prepares data from preset configuration (data_params, vectorizer_params) for model traning and testing.
    :param data_params: dict
    :param vectorizer_params: dict
    :return: dict. Contains cleaned texts (train/test) and also vectorized, ready for model input.
    """
    extractor_params = data_params['extraction_params']
    if data_params['use_cache'] and os.path.exists(f"{data_params['cache_dir']}/{CACHE_FILE_NAME}"):
        print('Retrieving data from cache...')
        return read_pickle(f"{data_params['cache_dir']}/{CACHE_FILE_NAME}")
    # retrieve data
    train_corpus, test_corpus = load_data(**extractor_params)

    corpus_word_limit = data_params['corpus_word_limit']

    if corpus_word_limit is not None:
        train_corpus = cut_off_longer_texts(train_corpus, 'text', corpus_word_limit)
        test_corpus = cut_off_longer_texts(test_corpus, 'text', corpus_word_limit)

    if data_params['use_corpus_balancing']:

        train_corpus = undersample_to_even(train_corpus, 'label')
        test_corpus = undersample_to_even(test_corpus, 'label')

    # prepare components for data processing
    data_cleaner = prepare_data_cleaner(data_params['cleaning_params'])
    vectorizer_func = vectorizer_params.get('vectorizer_retriever_func')
    data_vectorizer = vectorizer_func(vectorizer_params)
    preprocessor = DataPreprocessor(data_cleaner, data_vectorizer)

    # prepare data using prepared components
    train_corpus = preprocessor.clean(train_corpus)
    test_corpus = preprocessor.clean(test_corpus)

    preprocessor.fit(train_corpus)
    preprocessor.save(vectorizer_params['save_dir'])

    train_corpus_vec = preprocessor.vectorize(train_corpus)
    test_corpus_vec = preprocessor.vectorize(test_corpus)

    data = {
        'train_vectorized': train_corpus_vec,
        'test_vectorized': test_corpus_vec,
        'train_cleaned': train_corpus,
        'test_cleaned': test_corpus
    }
    if data_params['use_cache']:
        save_cache_data(data_params['cache_dir'], data)
    return data


def save_cache_data(save_dir, data):
    os.makedirs(save_dir, exist_ok=True)
    write_pickle(f'{save_dir}/{CACHE_FILE_NAME}', data)
    print(f'Cache data saved to {save_dir}/{CACHE_FILE_NAME}')


def prepare_data_cleaner(cleaning_params):
    """Prepares DataCleaner"""
    text_cleaner = TextCleaner(**cleaning_params['text'])
    output_cleaner = OutputCleaner(verifier_func=cleaning_params['output']['output_verification_func'])
    return PresetDataCleaner(text_cleaner, output_cleaner)
