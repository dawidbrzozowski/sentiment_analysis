import re
from keras import layers

"""
This module provides private functions for creating custom layers from descriptions.
These functions are not meant to be used for any other reasons.
"""

ACTIVATIONS = ['relu', 'softmax', 'tanh', 'sigmoid']

ACTIVATION_NAMES = '\n'.join([activation_name for activation_name in ACTIVATIONS])


class ActivationNotFoundError(Exception):
    pass


class UnitError(Exception):
    pass


class RateError(Exception):
    pass


class LayerNotFoundError(Exception):
    pass


def _build_maxpooling1d(layer_description):
    return layers.MaxPooling1D()


def _build_averagepooling1d(layer_description):
    return layers.AveragePooling1D()


def _build_globalmaxpooling1d(layer_description):
    return layers.GlobalMaxPooling1D()


def _build_globalaveragepooling1d(layer_description):
    return layers.GlobalMaxPooling1D()


def _build_lstm(layer_description: str):
    activation, units = _get_activation_and_units(layer_description)
    return_sequences = True if 'return_sequences' in layer_description else False
    return layers.LSTM(activation=activation, units=units, return_sequences=return_sequences)


def _build_gru(layer_description):
    activation, units = _get_activation_and_units(layer_description)
    return_sequences = True if 'return_sequences' in layer_description else False
    return layers.GRU(activation=activation, units=units, return_sequences=return_sequences)


def _build_dense(layer_description):
    activation, units = _get_activation_and_units(layer_description)
    return layers.Dense(units=units, activation=activation)


def _build_bidirectional(layer_description):
    layer_description = re.sub('bidirectional', '', layer_description)
    next_layer = None
    for layer_name in LAYER_BUILDERS:
        if layer_name in layer_description:
            next_layer = LAYER_BUILDERS[layer_name](layer_description)
            break
    if next_layer is None:
        raise LayerNotFoundError(
            f'Layer not found. Please include in the description one of the following layers:\n{LAYER_BUILDERS}')

    return layers.Bidirectional(next_layer)


def _build_dropout(layer_description):
    rate = _get_rate(layer_description)
    return layers.Dropout(rate=rate)


def _build_spatialdropout1d(layer_description):
    rate = _get_rate(layer_description)
    return layers.SpatialDropout1D(rate=rate)


def _build_flatten(layer_description):
    return layers.Flatten()


def _get_activation_and_units(layer_description):
    activation = None
    for activation_name in ACTIVATIONS:
        if activation_name in layer_description:
            activation = activation_name
            break
    if activation is None:
        raise ActivationNotFoundError(
            f'Activation function not found in layer description. Please try one of the following:\n{ACTIVATION_NAMES}')

    units = re.findall(r'\d+', layer_description)
    if not units:
        raise UnitError('Units amount should be provided')
    elif len(units) > 1:
        raise UnitError('One number for units should be provided')
    units = int(units[0])
    return activation, units


def _get_rate(layer_description):
    rates = re.findall(r'0\.\d+', layer_description)
    if not rates:
        raise RateError('Rate not found in layer description. Please include dropout rate.')
    elif len(rates) > 1:
        raise RateError('Too many rates found. Please include one dropout rate.')
    return float(rates[0])


LAYER_BUILDERS = {
    'bidirectional': _build_bidirectional,
    'dense': _build_dense,
    'globalmaxpooling1d': _build_globalmaxpooling1d,
    'maxpooling1d': _build_maxpooling1d,
    'globalaveragepooling1d': _build_globalaveragepooling1d,
    'averagepooling1d': _build_averagepooling1d,
    'lstm': _build_lstm,
    'gru': _build_gru,
    'spatialdropout1d': _build_spatialdropout1d,
    'dropout': _build_dropout,
    'flatten': _build_flatten,
}
