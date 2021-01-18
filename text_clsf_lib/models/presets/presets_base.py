from keras import layers

from text_clsf_lib.models.embedding.components_preparation import prepare_embedding_data_vectorizer
from text_clsf_lib.models.build.model_builder import TfIdfFFModelBuilder, EmbeddingFFModelBuilder, EmbeddingRNNModelBuilder
from text_clsf_lib.models.tfidf.components_preparation import prepare_tfidf_data_vectorizer
from text_clsf_lib.preprocessing.cleaning.data_cleaners import binary_output
from text_clsf_lib.preprocessing.vectorization.embeddings.embedding_loaders import WordEmbeddingsLoader
from text_clsf_lib.preprocessing.vectorization.output_vectorizers import BasicOutputVectorizer
from text_clsf_lib.preprocessing.vectorization.text_vectorizers import TfIdfTextVectorizer, \
    GloveEmbeddingTextVectorizer, \
    BagOfWordsTextVectorizer, BPEEmbeddingTextVectorizer

PRESETS = {

    'tfidf_feedforward': {
        'model_builder_class':                      TfIdfFFModelBuilder,
        'model_name':                              'nn_tfidf',
        'model_save_dir':                          '_models',
        'data_params': {
            'extraction_params': {
                'X_name':                           None,
                'y_name':                           None,
                'path':                             None,
                'test_size':                        None,
                'random_state':                     42,
            },
            'cache_dir':                            None,
            'use_cache':                            False,
            'use_corpus_balancing':                 False,
            'corpus_word_limit':                    None,
            'cleaning_params': {
                'text': {
                    'replace_numbers':              False,
                    'use_ner':                      False,
                    'use_ner_converter':            False,
                    'use_stemming':                 False,
                    'use_lemmatization':            False,
                    'use_twitter_data_preprocessing':False,
                    'lowercase':                    True
                },
                'output': {
                    'output_verification_func':     binary_output

                }
            },
        },
        'vectorizer_params': {
            'vectorizer_retriever_func':            prepare_tfidf_data_vectorizer,
            'save_dir':                            'preprocessor',
            'vector_width':                         1000,
            'text_vectorizer':                      TfIdfTextVectorizer,
            'output_vectorizer':                    BasicOutputVectorizer
        },

        'architecture_params': {
            'hidden_layers_list':                   [],
            'hidden_layers':                        2,
            'hidden_units':                         32,
            'hidden_activation':                   'relu',
            'output_activation':                   'softmax',
            'optimizer':                           'adam',
            'loss':                                'binary_crossentropy',
            'lr':                                   0.01,
            'metrics':                              ['accuracy'],
            'output_units':                         2
        },
        'training_params': {
            'epochs':                               1,
            'batch_size':                           128,
            'validation_split':                     0.1,
            'callbacks':                            None
        }
    },
    'bag_of_words_feedforward': {
        'model_builder_class':                      TfIdfFFModelBuilder,
        'model_name':                              'nn_tfidf',
        'model_save_dir':                          '_models',
        'data_params': {
            'extraction_params': {
                'X_name':                           None,
                'y_name':                           None,
                'path':                             None,
                'test_size':                        None,
                'random_state':                     42,
            },
            'cache_dir':                            None,
            'use_cache':                            False,
            'use_corpus_balancing':                 False,
            'corpus_word_limit':                    None,

            'cleaning_params': {
                'text': {
                    'replace_numbers':              False,
                    'use_ner':                      False,
                    'use_ner_converter':            False,
                    'use_stemming':                 False,
                    'use_lemmatization':            False,
                    'use_twitter_data_preprocessing':False,
                    'lowercase':                    True
                },
                'output': {
                    'output_verification_func':     binary_output

                }
            },
        },
        'vectorizer_params': {
            'vectorizer_retriever_func':            prepare_tfidf_data_vectorizer,
            'save_dir':                            'preprocessor',
            'vector_width':                         1000,
            'text_vectorizer':                      BagOfWordsTextVectorizer,
            'output_vectorizer':                    BasicOutputVectorizer
        },

        'architecture_params': {
            'hidden_layers_list':                   [],
            'hidden_layers':                        2,
            'hidden_units':                         32,
            'hidden_activation':                   'relu',
            'output_activation':                   'softmax',
            'optimizer':                           'adam',
            'loss':                                'binary_crossentropy',
            'lr':                                   0.01,
            'metrics':                              ['accuracy'],
            'output_units':                         2
        },
        'training_params': {
            'epochs':                               1,
            'batch_size':                           128,
            'validation_split':                     0.1,
            'callbacks':                            None
        }
    },

    'glove_feedforward': {
        'model_builder_class':                      EmbeddingFFModelBuilder,
        'model_name':                              'nn_embedding',
        'model_save_dir':                          '_models',
        'data_params': {
            'extraction_params': {
                'X_name':                           None,
                'y_name':                           None,
                'path':                             None,
                'test_size':                        None,
                'random_state':                     42,
            },
            'cache_dir':                            None,
            'use_cache':                            False,
            'use_corpus_balancing':                 False,
            'corpus_word_limit':                    None,

            'cleaning_params': {
                'text': {
                    'replace_numbers':              False,
                    'use_ner':                      False,
                    'use_ner_converter':            False,
                    'use_stemming':                 False,
                    'use_lemmatization':            False,
                    'use_twitter_data_preprocessing':False,
                    'lowercase':                    True
                },
                'output': {
                    'output_verification_func':     binary_output

                }
            }
        },
        'vectorizer_params': {
            'vectorizer_retriever_func':            prepare_embedding_data_vectorizer,
            'text_vectorizer':                      GloveEmbeddingTextVectorizer,
            'embeddings_loader':                    WordEmbeddingsLoader,
            'embedding_type':                      'glove_wiki',
            'output_vectorizer':                    BasicOutputVectorizer,
            'max_vocab_size':                       5000,
            'max_seq_len':                          25,
            'embedding_dim':                        50,
            'save_dir':                            'preprocessor',
        },

        'architecture_params': {
            'hidden_layers_list':                   [],
            'dimension_reducer':                    layers.Flatten,
            'hidden_layers':                        2,
            'hidden_units':                         32,
            'hidden_activation':                   'relu',
            'output_activation':                   'softmax',
            'optimizer':                           'adam',
            'loss':                                'binary_crossentropy',
            'lr':                                   0.01,
            'metrics':                              ['accuracy'],
            'trainable_embedding':                  False,
            'output_units':                         2
        },
        'training_params': {
            'epochs':                               5,
            'batch_size':                           128,
            'validation_split':                     0.1,
            'callbacks':                            None
        }
    },

    'glove_rnn': {
        'model_builder_class':                      EmbeddingRNNModelBuilder,
        'model_name':                              'nn_embedding',
        'model_save_dir':                          '_models',
        'data_params': {
            'extraction_params': {
                'X_name':                           None,
                'y_name':                           None,
                'path':                             None,
                'test_size':                        None,
                'random_state':                     42,
             },
            'cache_dir':                            None,
            'use_cache':                            False,
            'use_corpus_balancing':                 False,
            'corpus_word_limit':                    None,

            'cleaning_params': {
                'text': {
                    'replace_numbers':              False,
                    'use_ner':                      False,
                    'use_ner_converter':            False,
                    'use_stemming':                 False,
                    'use_lemmatization':            False,
                    'use_twitter_data_preprocessing':False,
                    'lowercase':                    True
                },
                'output': {
                    'output_verification_func':     binary_output
                }
            }
        },
        'vectorizer_params': {
            'vectorizer_retriever_func':            prepare_embedding_data_vectorizer,
            'text_vectorizer':                      GloveEmbeddingTextVectorizer,
            'embeddings_loader':                    WordEmbeddingsLoader,
            'embedding_type':                      'glove_twitter',
            'output_vectorizer':                    BasicOutputVectorizer,
            'max_vocab_size':                       5000,
            'max_seq_len':                          200,
            'embedding_dim':                        50,
            'save_dir':                            'preprocessor',

        },
        'architecture_params': {
            'hidden_layers_list':                   [],
            'hidden_layers':                        2,
            'hidden_units':                         32,
            'hidden_activation':                   'relu',
            'output_activation':                   'softmax',
            'optimizer':                           'adam',
            'loss':                                'binary_crossentropy',
            'lr':                                   0.01,
            'metrics':                              ['accuracy'],
            'trainable_embedding':                  False,
            'output_units':                         2
        },
        'training_params': {
            'epochs':                               2,
            'batch_size':                           128,
            'validation_split':                     0.1,
            'callbacks':                            None
        }
    },
    'bpe_rnn': {
        'model_builder_class':                      EmbeddingRNNModelBuilder,
        'model_name':                              'nn_embedding',
        'model_save_dir':                          '_models',
        'data_params': {
            'extraction_params': {
                'X_name':                           None,
                'y_name':                           None,
                'path':                             None,
                'test_size':                        None,
                'random_state':                     42,
            },
            'cache_dir':                            None,
            'use_cache':                            False,
            'use_corpus_balancing':                 False,
            'corpus_word_limit':                    None,

            'cleaning_params': {
                'text': {
                    'replace_numbers':              False,
                    'use_ner':                      False,
                    'use_ner_converter':            False,
                    'use_stemming':                 False,
                    'use_lemmatization':            False,
                    'use_twitter_data_preprocessing':False,
                    'lowercase':                    True
                },
                'output': {
                    'output_verification_func':     binary_output
                }
            }
        },
        'vectorizer_params': {
            'vectorizer_retriever_func':            prepare_embedding_data_vectorizer,
            'text_vectorizer':                      BPEEmbeddingTextVectorizer,
            'embedding_type':                      'bpe',
            'output_vectorizer':                    BasicOutputVectorizer,
            'max_vocab_size':                       5000,
            'max_seq_len':                          167,
            'embedding_dim':                        50,
            'save_dir':                            'preprocessor',

        },
        'architecture_params': {
            'hidden_layers_list':                   [],
            'hidden_layers':                        2,
            'hidden_units':                         32,
            'hidden_activation':                   'relu',
            'output_activation':                   'softmax',
            'optimizer':                           'adam',
            'loss':                                'binary_crossentropy',
            'lr':                                   0.01,
            'metrics':                              ['accuracy'],
            'trainable_embedding':                  False,
            'output_units':                         2
        },
        'training_params': {
            'epochs':                               2,
            'batch_size':                           128,
            'validation_split':                     0.1,
            'callbacks':                            None
        }
    }

}