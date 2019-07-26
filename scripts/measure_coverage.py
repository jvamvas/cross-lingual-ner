"""
Compute four different subword coverage measures between a two CoNLL datasets
"""

import logging
import sys
from collections import Counter
from copy import deepcopy

from pytorch_pretrained_bert import BertTokenizer

from .conll_statistics import CoNLL2003Dataset


class OverlapMeasure:

    def __init__(self, source: CoNLL2003Dataset, target: CoNLL2003Dataset, tokenizer: BertTokenizer):
        self.source = deepcopy(source)
        self.target = deepcopy(target)
        self.tokenizer = tokenizer

        self.source_words, self.source_names = self.tokenize(self.source)
        self.target_words, self.target_names = self.tokenize(self.target)

    def tokenize(self, dataset):
        words = []
        names = []
        for document in dataset.documents:
            for sentence in document:
                for token, ner_tag in sentence:
                    if token is None:
                        continue
                    word_pieces = self.tokenizer.tokenize(token)
                    words += word_pieces
                    if ner_tag != "O":
                        names += word_pieces
        return words, names

    def _type_coverage(self, list1, list2):
        return len(set(list1) & set(list2)) / len(set(list2))

    def _token_coverage(self, list1, list2):
        target_counter = Counter(list2)
        observed_set = set(list1) & set(list2)
        observed_count = 0
        for observed_type in observed_set:
            observed_count += target_counter[observed_type]
        return observed_count / len(list2)

    def get_word_type_coverage(self):
        return self._type_coverage(self.source_words, self.target_words)

    def get_word_token_coverage(self):
        return self._token_coverage(self.source_words, self.target_words)

    def get_name_type_coverage(self):
        return self._type_coverage(self.source_names, self.target_names)

    def get_name_token_coverage(self):
        return self._token_coverage(self.source_names, self.target_names)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    source = CoNLL2003Dataset(sys.argv[1])
    target = CoNLL2003Dataset(sys.argv[2])
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
    overlap_measure = OverlapMeasure(source, target, tokenizer)
    print(overlap_measure.get_word_type_coverage())
    print(overlap_measure.get_word_token_coverage())
    print(overlap_measure.get_name_type_coverage())
    print(overlap_measure.get_name_token_coverage())
