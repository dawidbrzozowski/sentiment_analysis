from text_clsf_lib.models.presets.presets_base import PRESETS
from text_clsf_lib.utils.files_io import write_json, load_json
import os

PRESET_FILE_NAME = 'preset_config.json'


def create_preset(
        # meta inf parameters
        preset_base: str,
        model_name: str,
        model_save_dir: str = None,
        # data parameters
        data_path=None,
        test_size=None,
        use_cache=None,
        use_corpus_balancing=None,
        corpus_word_limit=None,
        X_name=None,
        y_name=None,
        train_test_random_state=None,
        ner_cleaning: bool = None,
        ner_converter: bool = None,
        twitter_preprocessing: bool = None,
        replace_numbers: bool = None,
        use_lemmatization: bool = None,
        use_stemming: bool = None,
        use_lowercase: bool = None,
        output_verification_func=None,
        # vectorization parameters
        vector_width: int = None,
        embedding_type: str = None,
        max_vocab_size: int = None,
        max_seq_len: int = None,
        embedding_dim: int = None,
        preprocessor_save_dir: str = None,
        # architecture_parameters
        hidden_layers: int = None,
        hidden_layers_list: list = None,
        hidden_units: int = None,
        hidden_activation: str = None,
        output_activation: str = None,
        optimizer: str = None,
        loss: str = None,
        lr: float = None,
        metrics: list = None,
        output_units: str = None,
        # training params
        epochs: int = None,
        batch_size: int = None,
        validation_split: float = None,
        callbacks: list = None):
    """
    This function creates a preset for a custom model architecture.
    :param train_test_random_state: random state for train/test split for single file extraction.
    :param preset_base: Each model should be based on a base preset.
        This helps to speed up the process of model creation.
        Currently implemented preset_base:
            - tfidf_feedforward : use if you want your model to be based on tfidf vectorizer.
            - glove_feedforward : use if you want your model to be based on embeddings but not RNN.
            - glove_rnn : use if you want your model to be based on embeddings and RNN architecture.
    :param model_name: this will be the name of your model and also the directory in which it will be stored.
    :param model_save_dir: model parent directory. _models for default.
    :param data_path: str or tuple of two strings (train path, test path). If one provided, train_test_split will be executed.
    :param use_cache: if set to true, it will try to store cached processed data, to test faster later on.
    :param test_size: float if one data_path is provided, train test split will be executed with given test_size. Default 0.2.
    :param use_corpus_balancing: bool. Define if you want your samples to be even in terms of categories (using undersampling method).
    :param y_name: str. if use_corpus_balancing is used, you must provide key name for your y label in the corpus.
    :param corpus_word_limit: int. Define if you want to get rid of samples, that have more than corpus_word_limit words.
    :param X_name: str. If corpus_word_limit is used, you must provide name for the key for your X input texts.
    :param ner_cleaning: whether to use or not NER (Named Entity Recognition) preprocessor. This NER comes from SpaCy.
    It is not recommended for large dataset, since this might take a long time.
    :param ner_converter: When ner_cleaning is set to True, NER converter translates the names for better emedding understanding.
    :param twitter_preprocessing: When set to True, it will run twitter data preprocessing.
    :param replace_numbers: bool Replaces numbers in corpus with a special token.
    :param use_lemmatization: bool Lemmatizes corpus.
    :param use_stemming: bool Performs stemming on corpus.
    :param use_lowercase: bool: bool Performs lowercasing on corpus.
    Recommended especially when using GloVe Twitter embeddings.
    :param output_verification_func: provide your own function, for data verification.
        This function should check if the model output is correct.
    :param vector_width: Used for Tfidf vectorizer if used.
    :param embedding_type: str. Used for defining embedding type that is used. Choices: [bpe, glove_twitter, glove_wiki]
    :param embedding_dim: int. Define embedding dimension if embedding vectorizer is used.
    :param max_vocab_size: int. Define max vocab size for your embeddings if embedding vectorizer is used.
    :param max_seq_len: int. Define max sequence length for your embeddings if embedding vectorizer is used.
                Padding used.
    :param preprocessor_save_dir: if you want to have a custom preprocessor_save_dir provide it here.
    :param hidden_layers_list: If you want to create your model architecture using layer descriptions,
        provide layer descriptions here.
    :param hidden_layers: If layer descriptions is not used, basic model architecture will be used.
        Here you can provide amount of hidden layers.
    :param hidden_units: If layer descriptions is not used, basic model architecture will be used.
        Here you can provide amount of hidden units.
    :param hidden_activation: If layer descriptions is not used, basic model architecture will be used.
        Here you can provide hidden layer activation.
    :param output_activation: If layer descriptions is not used, basic model architecture will be used.
        Here you can provide output activation.
    :param optimizer: Provide your custom optimizer. Recommended usage of Keras optimizers.
    :param loss: Provide your custom loss. Recommended usage of Keras loss.
    :param lr: float. Pick your learning rate.
    :param metrics: Pick your metrics from Keras backend.
    :param output_units: If layer descriptions is not used, basic model architecture will be used.
        Here you can provide amount of output units.
    :param epochs: int.
    :param batch_size: int.
    :param validation_split: float.
    :param callbacks: Provide callbacks for your model. Recommended usage of Keras callbacks.
    :return: dict. Model preset.

    """
    args = locals()
    preset = dict(PRESETS[preset_base])
    _put_or_default(preset, model_name, '', 'model_name')
    model_save_dir = model_save_dir if model_save_dir is not None else preset['model_save_dir']
    _put_or_default(preset, f'{model_save_dir}/{model_name}/model', '', 'model_save_dir')
    _put_or_default(preset, data_path, 'data_params', 'data_extractor')
    _put_or_default(preset, use_corpus_balancing, 'data_params', 'use_corpus_balancing')
    _put_or_default(preset, corpus_word_limit, 'data_params', 'corpus_word_limit')
    _put_or_default(preset, X_name, 'data_params:extraction_params', 'X_name')
    _put_or_default(preset, y_name, 'data_params:extraction_params', 'y_name')
    _put_or_default(preset, data_path, 'data_params:extraction_params', 'path')
    _put_or_default(preset, f'{model_save_dir}/{model_name}/_cache', 'data_params', 'cache_dir')
    _put_or_default(preset, use_cache, 'data_params', 'use_cache')
    _put_or_default(preset, test_size, 'data_params:extraction_params', 'test_size')
    _put_or_default(preset, ner_cleaning, 'data_params:cleaning_params:text', 'use_ner')
    _put_or_default(preset, ner_converter, 'data_params:cleaning_params:text', 'use_ner_converter')
    _put_or_default(preset, twitter_preprocessing, 'data_params:cleaning_params:text', 'use_twitter_data_preprocessing')
    _put_or_default(preset, replace_numbers, 'data_params:cleaning_params:text', 'replace_numbers')
    _put_or_default(preset, use_lemmatization, 'data_params:cleaning_params:text', 'use_lemmatization')
    _put_or_default(preset, use_lowercase, 'data_params:cleaning_params:text', 'lowercase')
    _put_or_default(preset, use_stemming, 'data_params:cleaning_params:text', 'use_stemming')
    _put_or_default(preset, output_verification_func, 'data_params:cleaning_params:output', 'output_verification_func')
    _put_or_default(preset, vector_width, 'vectorizer_params', 'vector_width')
    _put_or_default(preset, embedding_type, 'vectorizer_params', 'embedding_type')
    _put_or_default(preset, max_vocab_size, 'vectorizer_params', 'max_vocab_size')
    _put_or_default(preset, max_seq_len, 'vectorizer_params', 'max_seq_len')
    _put_or_default(preset, embedding_dim, 'vectorizer_params', 'embedding_dim')
    _put_or_default(preset, preprocessor_save_dir, 'vectorizer_params', 'save_dir')
    preprocessor_save_dir = preprocessor_save_dir if preprocessor_save_dir is not None else preset['vectorizer_params']['save_dir']
    _put_or_default(preset, f'{model_save_dir}/{model_name}/{preprocessor_save_dir}', 'vectorizer_params', 'save_dir')
    _put_or_default(preset, hidden_layers, 'architecture_params', 'hidden_layers')
    _put_or_default(preset, hidden_layers_list, 'architecture_params', 'hidden_layers_list')
    _put_or_default(preset, hidden_units, 'architecture_params', 'hidden_units')
    _put_or_default(preset, hidden_activation, 'architecture_params', 'hidden_activation')
    _put_or_default(preset, output_activation, 'architecture_params', 'output_activation')
    _put_or_default(preset, optimizer, 'architecture_params', 'optimizer')
    _put_or_default(preset, loss, 'architecture_params', 'loss')
    _put_or_default(preset, lr, 'architecture_params', 'lr')
    _put_or_default(preset, metrics, 'architecture_params', 'metrics')
    _put_or_default(preset, output_units, 'architecture_params', 'output_units')
    _put_or_default(preset, epochs, 'training_params', 'epochs')
    _put_or_default(preset, batch_size, 'training_params', 'batch_size')
    _put_or_default(preset, validation_split, 'training_params', 'validation_split')
    _put_or_default(preset, callbacks, 'training_params', 'callbacks')

    save_preset(f'{model_save_dir}/{model_name}', args)

    return preset


def _put_or_default(preset: dict, value, context_path: str, attribute_name: str):
    if value is None:
        return
    dict_path_list = context_path.split(':')
    context = preset
    for el in dict_path_list:
        if el:
            context = context.get(el)
    if attribute_name in context.keys():
        context[attribute_name] = value


def load_preset(model_name: str, model_dir='_models'):
    preset_path = f'{model_dir}/{model_name}/{PRESET_FILE_NAME}'
    preset_args = load_json(preset_path)
    return create_preset(**preset_args)


def save_preset(save_dir: str, preset_args: dict):
    not_null_preset_args = {k: v for k, v in preset_args.items() if v is not None}
    os.makedirs(save_dir, exist_ok=True)
    write_json(f'{save_dir}/{PRESET_FILE_NAME}', not_null_preset_args)

