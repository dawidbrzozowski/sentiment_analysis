from abc import abstractmethod
from typing import List, Tuple

from text_clsf_lib.utils.files_io import load_json
from sklearn.model_selection import train_test_split


def load_data(path: str or tuple,
              X_name: str,
              y_name: str,
              test_size: float = 0.2,
              random_state=None):
    if type(path) is str:
        assert test_size is not None, 'Test size must be provided for split!'
        path_extension = path.split('.')[-1]
        if path_extension == 'json':
            data_extractor = SingleJsonDataExtractor()
            return data_extractor.get_train_test_corpus(corpus_path=path, test_size=test_size,
                                                        X_name=X_name, y_name=y_name, random_state=random_state)
        else:
            raise Exception('Data should be in json format!')
        # jeden zbior i trzeba podzielic
        pass
    elif type(path) in (tuple, list) and len(path) == 2:
        train_path, test_path = path
        data_extractor = DoubleJsonDataExtractor()
        return data_extractor.get_train_test_corpus(train_path=train_path, test_path=test_path,
                                                    X_name=X_name, y_name=y_name)
    else:
        raise Exception('One path or a tuple of two paths should be given!')


class DataExtractor:
    """
    Base class for retrieving train test sets.
    """

    @abstractmethod
    def get_train_test_corpus(self, **kwargs) -> Tuple[List, List]:
        pass


class DoubleJsonDataExtractor(DataExtractor):
    """
    This class is meant to retrieve data from custom jsons (train and test separately).

    The provided Json must be a list of dict containing fields such as:
    - 'text': str
    - 'label' (0 or 1) OR (true or false)
    """

    def get_train_test_corpus(self, train_path: str, test_path: str,
                              X_name: str, y_name: str) -> Tuple[List, List]:
        """
        :param custom_sample_func: if your data needs any special processing - add this function here.
        :param X_name: name for X in your corpus.
        :param y_name: name for y in your corpus.
        :param train_path: path to train set.
        :param test_path: path to test set
        :return: List[dict] train, List[dict] test.
        """
        assert train_path is not None and test_path is not None and X_name is not None and y_name is not None, \
            "Can't extract data, when not all arguments are provided."

        data_train = load_json(train_path)
        data_test = load_json(test_path)

        data_train = [swap_key_name(sample, X_name, 'text') for sample in data_train]
        data_train = [swap_key_name(sample, y_name, 'label') for sample in data_train]
        data_test = [swap_key_name(sample, X_name, 'text') for sample in data_test]
        data_test = [swap_key_name(sample, y_name, 'label') for sample in data_test]

        return data_train, data_test


class SingleJsonDataExtractor(DataExtractor):
    """
    This class is meant to retrieve data from custom json (train and test together).

    The provided Json must be a list of dict containing fields such as:
    - 'text': str
    - 'label' (0 or 1) OR (true or false)
    """

    def get_train_test_corpus(self, corpus_path: str, test_size: float,
                              X_name: str, y_name: str, random_state: int = 42) -> Tuple[List, List]:
        """
        :param random_state: int for train test split
        :param X_name: name for X in your corpus.
        :param y_name: name for y in your corpus.
        :param corpus_path: Path to corpus.
        :param custom_sample_func: if your data needs any special processing - add this function here (on sample level).
        :param test_size: float
        :return: List[dict] train, List[dict] test.
        """

        assert X_name is not None and y_name is not None and corpus_path is not None and test_size is not None, \
            "Can't extract data, when not all arguments are provided."
        corpus = load_json(corpus_path)

        corpus = [swap_key_name(sample, X_name, 'text') for sample in corpus]
        corpus = [swap_key_name(sample, y_name, 'label') for sample in corpus]
        stratify_column = [sample['label'] for sample in corpus]
        data_train, data_test = train_test_split(
            corpus, stratify=stratify_column, test_size=test_size, random_state=random_state)
        return data_train, data_test


def swap_key_name(sample: dict, src_name: str, dest_name: str):
    if src_name != dest_name:
        sample[dest_name] = sample[src_name]
        del sample[src_name]
    return sample
