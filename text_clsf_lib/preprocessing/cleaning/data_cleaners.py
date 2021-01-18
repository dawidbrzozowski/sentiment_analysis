import re
from abc import abstractmethod
from typing import List, Tuple
import en_core_web_sm
from nltk import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from text_clsf_lib.utils.files_io import load_json, write_json
import os
NER_CONVERTER_DEF_PATH = 'text_clsf_lib/preprocessing/cleaning/resources/ner_converter.json'


class DataCleaner:
    """
    This class is meant to be responsible for cleaning text data and output data.
    It should remove damaged records and process the rest.
    """

    @abstractmethod
    def clean(self, data: List[dict]) -> Tuple[list, list]:
        pass

    @abstractmethod
    def save(self, save_dir):
        pass


class BaselineDataCleaner(DataCleaner):
    def save(self, save_dir):
        write_json(f'{save_dir}/predictor_config.json', {'text_cleaner': {}})

    def clean(self, data: List[dict]):
        return data


class TextCleaner:
    """
    This class allows text cleaning.
    has 4 main options:
    - replace_numbers: replaces numbers with special encoding <number> if set to True.
    - use_ner: uses NER (Named Entity Recognition) from SpaCy for feature extraction if set to True.
    - use_ner_converter: when use_ner is switched on, it will convert NER encodings to more proper version.
                    This is recommended mostly for embedding models, since it's better understood by embedding matrix.
    - use_twitter_data_preprocessing: preprocesses texts, so that it is easier for GloVe twitter embeddings
                                    to understand the text.
    - use_stemming: stem texts using PorterStemmer
    - use_lemmatization: lemmatize texts using WordNetLemmatizer
    """
    def __init__(self,
                 replace_numbers=False,
                 use_ner=False,
                 use_ner_converter=False,
                 use_stemming=False,
                 use_lemmatization=False,
                 use_twitter_data_preprocessing=False,
                 lowercase=True):
        self.replace_numbers = replace_numbers
        self.ner_tagger = en_core_web_sm.load() if use_ner else None
        self.ner_converter = None
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.use_twitter_data_preprocessing = use_twitter_data_preprocessing
        self.lowercase = lowercase
        if use_ner and use_ner_converter:
            self.ner_converter = load_json(NER_CONVERTER_DEF_PATH)

    def save(self, save_dir):
        text_cleaner_config = {
            'text_cleaner': {
                'replace_numbers': self.replace_numbers,
                'use_ner': True if self.ner_tagger is not None else False,
                'use_ner_converter': True if self.ner_converter is not None else False,
                'use_stemming': True if self.stemmer is not None else False,
                'use_lemmatization': True if self.lemmatizer is not None else False,
                'use_twitter_data_preprocessing': self.use_twitter_data_preprocessing,
                'lowercase': self.lowercase
                }
        }
        os.makedirs(f'{save_dir}', exist_ok=True)
        write_json(f'{save_dir}/predictor_config.json', text_cleaner_config)

    def clean(self, texts: List[str]):
        print('Started data cleaning...')
        if self.use_twitter_data_preprocessing:
            print('Twitter data cleaning...')
            texts = self._preprocess_twitter_data(texts)
        if self.ner_tagger:
            print('NER tagging...')
            texts = self._perform_ner_on_texts(texts)
        if self.replace_numbers:
            print('Replacing numbers...')
            texts = self._replace_numbers(texts)
        if self.stemmer is not None:
            print('Stemming text...')
            texts = self._stem_texts(texts)
        if self.lemmatizer is not None:
            print('Lemmatizing text...')
            texts = self._lemmatize_texts(texts)
        if self.lowercase:
            print('Lowercasing...')
            texts = self._lowercase_texts(texts)
        print('Data cleaning finished!')
        return texts

    def _lowercase_texts(self, texts):
        return [text.lower() for text in texts]

    def _perform_ner_on_texts(self, texts):
        processed_texts = []
        for text in texts:
            ents = self.ner_tagger(text).ents
            for ent in ents:
                convert = lambda label: label if self.ner_converter is None else self.ner_converter[label]
                text = text.replace(str(ent), convert(ent.label_))
            processed_texts.append(text)
        return processed_texts

    def _stem_texts(self, texts):
        stemmed_texts = []
        for text in texts:
            stemmed_texts.append(' '.join([self.stemmer.stem(word) for word in word_tokenize(text)]))
        return stemmed_texts

    def _lemmatize_texts(self, texts):
        lemmatized_texts = []
        for text in texts:
            lemmatized_texts.append(' '.join([self.lemmatizer.lemmatize(word) for word in word_tokenize(text)]))
        return lemmatized_texts

    def _replace_numbers(self, texts):
        number_pattern = re.compile(r'[-+]?[.\d]*[\d]+[:,.\d]*')
        return [number_pattern.sub('<number>', text) for text in texts]

    def _preprocess_twitter_data(self, texts):
        """ Inspired from https://nlp.stanford.edu/projects/glove/preprocess-twitter.rb"""
        # Different regex parts for smiley faces
        processed_texts = []
        for text in texts:
            eyes = "[8:=;]"
            nose = "['`-]?"
            text = re.sub(r"https?://\S+\b|www\.(\w+\.)+\S*", "<url>", text)
            text = re.sub("@\w+", "<user>", text)
            text = re.sub(f"{eyes}{nose}[)d]+|[)d]+{nose}{eyes}", "<smile>", text, flags=re.IGNORECASE)
            text = re.sub(f'{eyes}{nose}p+', '<lolface>', text, flags=re.IGNORECASE)
            text = re.sub(f'{eyes}{nose}\(+|\)+{nose}{eyes}', '<sadface>', text)
            text = re.sub(f'{eyes}{nose}[\\\/|l*]', '<neutralface>', text)
            text = re.sub("/", " / ", text)
            text = re.sub('<3', '<heart>', text)
            text = re.sub(f'[-+]?[.\d]*[\d]+[:,.\d]*', '<number>', text)
            text = re.sub(f'#(?=\w+)', '<hashtag> ', text)
            processed_texts.append(text)
        return processed_texts


def binary_output(output):
    return True if output in (0, 1) else False


class OutputCleaner:
    """
    Output cleaner will remove rows, for which output is invalid.
        Takes in verifier_func, which should allow to pass 1 argument
        and return boolean value based on the argument's correctness.
    """

    def __init__(self, verifier_func=None):
        self.verifier_func = verifier_func

    def clean(self, data: List[dict]):
        correct_data = []
        for sample in data:
            if self.verifier_func is None or self.verifier_func(sample['label']):
                correct_data.append(sample)
        return correct_data


class PresetDataCleaner(DataCleaner):
    """
    Uses TextCleaner and OutputCleaner to clean the whole dataset.
    """

    def save(self, save_dir):
        self.text_cleaner.save(save_dir)

    def __init__(self, text_cleaner: TextCleaner, output_cleaner: OutputCleaner):
        self.text_cleaner = text_cleaner
        self.output_cleaner = output_cleaner

    def clean(self, data: List[dict]) -> Tuple[list, list]:
        texts = [sample['text'] for sample in data]
        cleaned_texts = self.text_cleaner.clean(texts)
        cleaned_data = []
        for sample, cleaned_text in zip(data, cleaned_texts):
            cleaned_data.append({
                'text': cleaned_text,
                'label': sample['label']})
        cleaned_data = self.output_cleaner.clean(cleaned_data)
        cleaned_texts = [sample['text'] for sample in cleaned_data]
        cleaned_outputs = [sample['label'] for sample in cleaned_data]
        return cleaned_texts, cleaned_outputs

