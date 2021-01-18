from text_clsf_lib.preprocessing.vectorization.data_vectorizers import DataVectorizer


def prepare_tfidf_data_vectorizer(vectorizer_params):
    """
    Takes parameters from preset vectorizer_params and prepares custom DataVectorizer based on tfidf.
    :param vectorizer_params: dict
    :return: DataVectorizer based on tfidf architecture.
    """
    vector_width = vectorizer_params['vector_width']
    text_vectorizer = vectorizer_params['text_vectorizer'](vector_width)
    output_vectorizer = vectorizer_params['output_vectorizer']()
    return DataVectorizer(text_vectorizer, output_vectorizer)
