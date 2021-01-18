from text_clsf_lib.preprocessing.vectorization.data_vectorizers import DataVectorizer


def prepare_embedding_data_vectorizer(vectorizer_params):
    """
    Takes parameters from preset vectorizer_params and prepares custom DataVectorizer based on embedding.
    :param vectorizer_params: dict
    :return: DataVectorizer based on embedding architecture.
        def __init__(self, max_vocab_size, max_seq_len, embedding_dim, embedding_type):

    """
    text_vectorizer = vectorizer_params['text_vectorizer'](
        max_vocab_size=vectorizer_params['max_vocab_size'],
        max_seq_len=vectorizer_params['max_seq_len'],
        embedding_dim=vectorizer_params['embedding_dim'],
        embedding_type=vectorizer_params['embedding_type'])

    output_vectorizer = vectorizer_params['output_vectorizer']()
    return DataVectorizer(text_vectorizer, output_vectorizer)