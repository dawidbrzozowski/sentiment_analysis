from typing import List

from text_clsf_lib.models.build.layers_builder_utils import LAYER_BUILDERS, LayerNotFoundError

LAYER_NAMES = '\n'.join([layer_name for layer_name in LAYER_BUILDERS])


def build_layers(last_hidden_layer, layer_descriptions: List[str]):
    """
    Creates neural network architecture from layer descriptions.
    This function is meant to create everything after vectorization:
    - input in case of tfidf/bag of words
    - embedding layer in case of embeddings.

    Layer Description:
    - Layer description should be a string, which contains information needed to create a layer.

    Available layers:
    - 'dense'.
    Dense layer.
    Args: units amount, activation type. Sample usage: 'dense 12 relu'
    - 'lstm'.
    LSTM layer.
    Args: units amount, activation type, return_sequences.
    Sample usage: 'lstm 12 tanh return_sequences' or 'lstm 12 tanh' if we only care about last result.
    - 'gru'.
    GRU layer.
    Args: units amount, activation type, return_sequences.
    Sample usage: 'gru 12 tanh return_sequences' or 'gru 12 tanh' if we only care about last result.
    - 'bidirectional'
    Bidirectional layer. Add this keyword to your layer to make it bidirectional: Sample use:
    'bidirectional lstm 12 tanh'
    - 'globalmaxpooling1d', 'maxpooling1d', 'globalaveragepooling1d', 'averagepooling1d'
    Pooling 1d layers.
    Sample use: 'globalmaxpooling1d'. No arguments needed.
    - 'dropout', 'spatialdropout1d'
    Dropout layers. Rate of dropout must be provided.
    Sample use: 'dropout 0.1'
    - 'flatten'.
    Flatten layer. Use without arguments.

    Available activations:
    relu, tanh, softmax, sigmoid

    :param last_hidden_layer: embedding/input layer.
    :param layer_descriptions:  List[str]. Each string describes one layer.
    :return: last layer of your model architecture.
    """
    for layer_description in layer_descriptions:
        layer = None
        for layer_name in LAYER_BUILDERS:
            if layer_name in layer_description:
                layer = LAYER_BUILDERS[layer_name](layer_description)
                last_hidden_layer = layer(last_hidden_layer)
                break
        if layer is None:
            raise LayerNotFoundError(
                f'Layer not found. Please include in the description one of the following layers:\n{LAYER_BUILDERS}')
    return last_hidden_layer
