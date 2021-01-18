from text_clsf_lib.models.model_trainer_runner import NNModelRunner
from text_clsf_lib.preprocessing.cleaning.data_cleaners import TextCleaner
from text_clsf_lib.preprocessing.preprocessors import RealDataPreprocessor
from text_clsf_lib.preprocessing.vectorization.text_vectorizers import LoadedTfIdfTextVectorizer, \
    LoadedEmbeddingTextVectorizer, LoadedBPEEmbeddingTextVectorizer
from text_clsf_lib.utils.files_io import read_pickle


def get_embedding_preprocessor(preprocessing_params: dict):
    """
    Prepares embedding preprocessor for predictor from preset and saved preprocessing files during training.
    :param preprocessing_params: dict.
    :return: RealDataPreprocessor object.
    """

    vectorizer_params = preprocessing_params['vectorizer_params']
    model_dir = preprocessing_params['model_dir']
    predictor_config = preprocessing_params['predictor_config']
    text_cleaner_config = predictor_config['text_cleaner']
    text_cleaner = TextCleaner(**text_cleaner_config)
    tokenizer_path = f"{model_dir}/{vectorizer_params['text_encoder_path']}"
    tokenizer = read_pickle(tokenizer_path)
    text_vectorizer = LoadedEmbeddingTextVectorizer(
        predictor_config=predictor_config,
        tokenizer=tokenizer)

    return RealDataPreprocessor(
        text_cleaner=text_cleaner,
        loaded_text_vectorizer=text_vectorizer)


def get_bpe_preprocessor(preprocessing_params: dict):
    predictor_config = preprocessing_params["predictor_config"]
    text_cleaner_config = predictor_config['text_cleaner']
    text_cleaner = TextCleaner(**text_cleaner_config)
    text_vectorizer = LoadedBPEEmbeddingTextVectorizer(predictor_config=predictor_config)

    return RealDataPreprocessor(
        text_cleaner=text_cleaner,
        loaded_text_vectorizer=text_vectorizer)


def get_tfidf_preprocessor(preprocessing_params: dict):
    """
    Prepares tfidf preprocessor for predictor from preset and saved preprocessing files during traning.
    :param preprocessing_params: dict.
    :return: RealDataPreprocessor object.
    """
    vectorizer_params = preprocessing_params['vectorizer_params']
    model_dir = preprocessing_params['model_dir']
    predictor_config = preprocessing_params['predictor_config']
    vectorizer_path = f"{model_dir}/{vectorizer_params['vectorizer_path']}"
    vectorizer = read_pickle(vectorizer_path)
    text_vectorizer = LoadedTfIdfTextVectorizer(vectorizer=vectorizer)
    text_cleaner_config = predictor_config['text_cleaner']
    text_cleaner = TextCleaner(**text_cleaner_config)

    return RealDataPreprocessor(
        text_cleaner=text_cleaner,
        loaded_text_vectorizer=text_vectorizer)


def get_model(model_path: str):
    return NNModelRunner(model_path=model_path)
