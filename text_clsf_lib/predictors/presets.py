from text_clsf_lib.predictors.predictor_commons import get_embedding_preprocessor, get_model, get_tfidf_preprocessor, \
    get_bpe_preprocessor
from text_clsf_lib.utils.files_io import load_json

CONFIG_PATH = 'preprocessor/predictor_config.json'

PRESETS = {

    'count_predictor': {
        'preprocessor_func':                        get_tfidf_preprocessor,
        'model_func':                               get_model,
        'preprocessing_params': {
            'vectorizer_params': {
                'vectorizer_path':                 'preprocessor/vectorizer.vec',
            }
        }
    },

    'embedding_predictor': {
        'model_func':                               get_model,
        'preprocessor_func':                        get_embedding_preprocessor,
        'preprocessing_params': {
            'vectorizer_params': {
                'text_encoder_path':               'preprocessor/tokenizer.pickle',
            },
        }
    },

    'bpe_predictor': {
        'model_func':                               get_model,
        'preprocessor_func':                        get_bpe_preprocessor,
        'preprocessing_params': {}
    },
}


def create_predictor_preset(
        model_name: str,
        model_dir: str = '_models'):
    """
    This function should be used for preparing predictor preset.
    It uses base presets, that are overridden by values provided in arguments.
    :param model_name: str, Your model name. This will be used for determining where is your model located.
    :param model_dir: parent directory of a model. Default value: '_models'
     better undestood by embedding matrix.
    :return: dict, predictor preset.
    """
    predictor_presets = {
        'count': 'count_predictor',
        'word_embedding': 'embedding_predictor',
        'bpe_embedding': 'bpe_predictor'
    }
    model_dir = f'{model_dir}/{model_name}'
    predictor_config = load_json(f'{model_dir}/{CONFIG_PATH}')
    type_ = predictor_config['vectorizer']['type']
    assert type_ in predictor_presets, f'Type should be one of the following: {[k for k in predictor_presets]}'
    preset = PRESETS[predictor_presets[type_]]
    preset['preprocessing_params']['predictor_config'] = predictor_config
    model = f'{model_name}.h5'
    preset['preprocessing_params']['model_dir'] = model_dir
    preset['model_path'] = f'{model_dir}/model/{model}'
    return preset
