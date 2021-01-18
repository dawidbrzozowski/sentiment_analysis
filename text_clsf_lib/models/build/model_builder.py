from abc import abstractmethod
from keras import layers, Model
from keras.optimizers import Adam

from text_clsf_lib.models.build.layers_builder import build_layers
from text_clsf_lib.utils.files_io import read_numpy


class ModelBuilder:
    """
    Base class for creating model architecture.
    Takes in architecture and vectorizer params from the preset and creates a model ready to be trained.
    ModelBuilder is meant to create models for classification task.
    It is not recommended to use it for regression tasks.
    """
    def __init__(self, architecture_params, vectorizer_params):
        self.architecture_params = architecture_params
        self.vectorizer_params = vectorizer_params

    @abstractmethod
    def prepare_model_architecture(self):
        pass

    def prepare_output_layer(self):
        return layers.Dense(
            units=self.architecture_params['output_units'],
            activation=self.architecture_params['output_activation'])

    def create_model(self, input_layer, output_layer):
        model = Model(input_layer, output_layer)
        optimizer = Adam(lr=self.architecture_params['lr'])
        model.compile(
            optimizer=optimizer,
            loss=self.architecture_params['loss'],
            metrics=self.architecture_params['metrics'])
        model.summary()
        return model


class FFModelBuilder(ModelBuilder):
    """
    Base class for Feedforward architecture builders.
    One abstract method to be implemented:
    def prepare_input_layers(self).
    """
    def prepare_model_architecture(self):
        input_, emb_layer = self.prepare_input_layers()
        if emb_layer is None:
            emb_layer = input_
        hidden = emb_layer
        if self.architecture_params['hidden_layers_list']:
            output_layer = build_layers(hidden, self.architecture_params['hidden_layers_list'])
        else:
            for _ in range(self.architecture_params['hidden_layers']):
                hidden = layers.Dense(
                    units=self.architecture_params['hidden_units'],
                    activation=self.architecture_params['hidden_activation'])(hidden)
            output_layer = self.prepare_output_layer()(hidden)
        return self.create_model(input_, output_layer)

    @abstractmethod
    def prepare_input_layers(self):
        pass


class TfIdfFFModelBuilder(FFModelBuilder):
    def prepare_input_layers(self):
        return layers.Input(shape=(self.vectorizer_params['vector_width'],)), None


class EmbeddingModelBuilder(ModelBuilder):
    """
    Base class for embedding-oriented model architectures.
    Implements embedding_layer retrieval.
    """

    @abstractmethod
    def prepare_model_architecture(self):
        pass

    def get_embedding_layer(self):
        embedding_matrix = read_numpy(f'{self.vectorizer_params["save_dir"]}/embedding_matrix.npy')
        embedding_layer = layers.Embedding(
            input_dim=len(embedding_matrix),
            output_dim=self.vectorizer_params['embedding_dim'],
            weights=[embedding_matrix],
            input_length=self.vectorizer_params['max_seq_len'],
            trainable=self.architecture_params['trainable_embedding']
        )
        return embedding_layer


class EmbeddingFFModelBuilder(FFModelBuilder, EmbeddingModelBuilder):
    """
    Class implements preparation of input layers for feedforward embedding-based models.
    """
    def prepare_input_layers(self):
        input_ = layers.Input(shape=(self.vectorizer_params['max_seq_len'],))
        emb_layer = self.get_embedding_layer()(input_)
        return input_, self.architecture_params['dimension_reducer']()(emb_layer)


class EmbeddingRNNModelBuilder(EmbeddingModelBuilder):
    """
    Class implements preparation of architecture for embedding RNN models.
    """
    def prepare_model_architecture(self):
        input_, emb_layer = self.prepare_input_layers()
        hidden = emb_layer
        if self.architecture_params['hidden_layers_list']:
            output = build_layers(hidden, self.architecture_params['hidden_layers_list'])
        else:
            hidden = layers.Bidirectional(layers.LSTM(
                units=self.architecture_params['hidden_units'],
                return_sequences=True))(hidden)
            hidden = layers.GlobalMaxPooling1D()(hidden)
            output = layers.Dense(
                units=self.architecture_params['output_units'],
                activation=self.architecture_params['output_activation'])(hidden)
        return self.create_model(input_, output)

    def prepare_input_layers(self):
        input_ = layers.Input(shape=(self.vectorizer_params['max_seq_len'],))
        return input_, self.get_embedding_layer()(input_)
