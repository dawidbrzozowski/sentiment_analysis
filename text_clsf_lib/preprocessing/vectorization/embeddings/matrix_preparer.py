import numpy as np


class EmbeddingsMatrixPreparer:
    """
    This class is meant to prepare embedding matrix, which is basically idx2vector.
    Return shape: (vocab_size x embedding_dim).
    """
    def __init__(self, word2idx, word2vec):
        self.word2idx = word2idx
        self.word2vec = word2vec

    def prepare_embedding_matrix(self):
        print('Filling pre-trained embeddings.')
        num_words = len(self.word2idx) + 1
        embedding_dim = len(list(self.word2vec.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))
        for word, idx in self.word2idx.items():
            embedding_vector = self.word2vec.get(word)
            if embedding_vector is not None:
                embedding_matrix[idx] = embedding_vector
        return embedding_matrix
