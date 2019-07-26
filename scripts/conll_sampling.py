"""
Take a sample from a CoNLL-formatted dataset, with a given number of sentences.

The original order and the document structure are ignored.
The sampling is uniform, but if the resulting sample does not contain all tags at least once,
it is discarded and the sampling is repeated.
"""

import itertools
import random
import sys
from copy import deepcopy


class CoNLL2003Dataset:

    def __init__(self, filepath):
        self.documents = []
        with open(filepath, encoding="utf-8") as f:
            document = []
            sentence = []
            for line in f:
                if self._is_document_separator(line):
                    if document:
                        self.documents.append(document)
                        document = []
                elif self._is_sentence_separator(line):
                    if sentence:
                        document.append(sentence)
                        sentence = []
                elif self._is_token_line(line):
                    if " " in line or "\t" in line:
                        token, ner_tag = self._parse_token_line(line)
                    else:
                        token = None
                        ner_tag = line.strip()
                    sentence.append((token, ner_tag))
                else:
                    print("Could not parse line:\n{}".format(line))
            if sentence:
                document.append(sentence)
            if document:
                self.documents.append(document)

    def _is_document_separator(self, line):
        return "DOCSTART" in line

    def _is_sentence_separator(self, line):
        return not line.strip()

    def _is_token_line(self, line):
        return bool(line.strip())

    def _parse_token_line(self, line):
        fields = line.strip().split()
        return fields[0], fields[-1]

    def get_num_sentences(self):
        num_sentences = 0
        for document in self.documents:
            num_sentences += len(document)
        return num_sentences

    def get_subset(self, n):
        subset = deepcopy(self)
        all_sentences = list(itertools.chain.from_iterable(self.documents))
        sample_sentences = random.sample(all_sentences, n)
        sample_documents = [[sentence] for sentence in sample_sentences]
        subset.documents = sample_documents
        return subset

    def get_raw_classes(self):
        classes = set()
        for document in self.documents:
            for sentence in document:
                for _, tag in sentence:
                    classes.add(tag)
        return sorted(list(classes))

    def __str__(self):
        s = []
        for document in self.documents:
            s.append("-DOCSTART- O\n\n")
            for sentence in document:
                for word in sentence:
                    s.append("\t".join(word) + "\n")
                s.append("\n")
        return "".join(s)


if __name__ == "__main__":
    full_dataset = CoNLL2003Dataset(sys.argv[1])
    n = int(sys.argv[2])
    subset = full_dataset.get_subset(n)
    while len(subset.get_raw_classes()) != len(full_dataset.get_raw_classes()):
        subset = full_dataset.get_subset(n)
    print(subset)
