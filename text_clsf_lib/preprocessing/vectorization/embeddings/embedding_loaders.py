import numpy as np

EMBEDDINGS_DIR = 'text_clsf_lib/preprocessing/vectorization/resources/embeddings'

EMBEDDING_TYPES = {
    'glove_twitter': 'glove/twitter',
    'glove_wiki': 'glove/wiki'
}


class WordEmbeddingsLoader:
    """
    Loads in GloVe embeddings for twitter or wiki, depending on glove_type.
    """
    def __init__(self, embedding_type: str):
        assert  embedding_type in EMBEDDING_TYPES, \
            f'Embedding type should be one of the following: {[key for key in EMBEDDING_TYPES]}'
        self.embedding_dir = f'{EMBEDDINGS_DIR}/{EMBEDDING_TYPES[embedding_type]}'

    def load_word_vectors(self, embedding_dim) -> dict:
        print(f'Loading pre-trained GloVe word vectors from {self.embedding_dir}/{embedding_dim}d.txt.')
        word2vec = {}
        with open(f'{self.embedding_dir}/{embedding_dim}d.txt') as f:
            for line in f:
                values = line.split()
                word = values[0]
                vec = np.asarray(values[1:], dtype='float32')
                word2vec[word] = vec
        print(f'Found {len(word2vec)} vectors.')
        return word2vec
