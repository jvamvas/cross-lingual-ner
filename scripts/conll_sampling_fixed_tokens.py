"""
Take a sample from a CoNLL-formatted dataset that has a given number of sentences and approximates a given number of tokens.

The original order and the document structure are ignored.
The sampling is uniform, but if the resulting sample does not contain all tags at least once,
it is discarded and the sampling is repeated.
"""

import itertools
import random
import sys
from copy import deepcopy

from .conll_sampling import CoNLL2003Dataset

MAX_SEQ_LEN = 150


def _count_tokens(sentences):
    return sum([min(MAX_SEQ_LEN, len(sentence)) for sentence in sentences])


def get_subset(dataset: CoNLL2003Dataset, n: int, num_tokens: int) -> CoNLL2003Dataset:
    subset = deepcopy(dataset)
    all_sentences = list(itertools.chain.from_iterable(subset.documents))
    random.shuffle(all_sentences)
    all_sentences.sort(key=len, reverse=True)
    target_mean_tokens = num_tokens / n
    while True:
        diff = target_mean_tokens - (_count_tokens(all_sentences) / len(all_sentences))
        if abs(diff) < 0.001:
            break
        if diff > 0:
            for _ in range(20): all_sentences.pop()
        else:
            assert len(all_sentences) > 20
            all_sentences = all_sentences[20:]
    sample_sentences = random.sample(all_sentences, n)
    subset.documents = [[sentence] for sentence in sample_sentences]
    return subset


if __name__ == "__main__":
    full_dataset = CoNLL2003Dataset(sys.argv[1])
    n = int(sys.argv[2])
    num_tokens = int(sys.argv[3])
    subset = get_subset(full_dataset, n, num_tokens)
    while len(subset.get_raw_classes()) != len(full_dataset.get_raw_classes()):
        subset = get_subset(full_dataset, n, num_tokens)
    print(subset)
