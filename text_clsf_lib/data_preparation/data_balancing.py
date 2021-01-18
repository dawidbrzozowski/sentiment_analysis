from collections import defaultdict
from random import shuffle
from typing import List


def undersample_to_even(corpus: List[dict], y_label_name: str):
    per_category = defaultdict(list)
    for sample in corpus:
        per_category[sample[y_label_name]].append(sample)

    min_samples_in_category = min([len(per_category[category]) for category in per_category])
    print(f'The category with the least amount of samples contains: {min_samples_in_category} samples.')
    undersampled = []
    for category in per_category:
        per_category[category] = per_category[category][:min_samples_in_category]
        shuffle(per_category[category])
        undersampled.extend(per_category[category])
    print(f'Undersampling completed. Returning corpus containing {len(undersampled)} samples.')
    shuffle(undersampled)
    return undersampled


def cut_off_longer_texts(corpus: List[dict], x_name: str, word_limit):
    filtered_by_length = []
    for sample in corpus:
        word_count = len(sample[x_name].split())
        if word_count <= word_limit:
            filtered_by_length.append(sample)
    return filtered_by_length
