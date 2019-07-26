"""
Print a table with statistics for a given CoNLL dataset.

Example:
Statistics for data/conll03/valid.txt:

Number of documents     216
Number of sentences     3250
Number of tokens        51362
Vocabulary size 9966

Average length of documents     15.046296296296296
Shortest document       3
Longest document        168

Average length of sentences     15.803692307692307
Shortest sentence       1
Longest sentence        109
Number of classes       5

Class distribution:
O       42759
PER     1842
LOC     1837
ORG     1341
MISC    922
5942
"""

import sys
from collections import Counter, defaultdict

import numpy as np


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

    def get_num_tokens(self):
        num_tokens = 0
        for document in self.documents:
            for sentence in document:
                num_tokens += len(sentence)
        return num_tokens

    def get_num_sentences(self):
        num_sentences = 0
        for document in self.documents:
            num_sentences += len(document)
        return num_sentences

    def get_num_documents(self):
        return len(self.documents)

    def get_document_lengths(self):
        return [len(document) for document in self.documents]

    def get_document_token_counts(self):
        return [sum([len(sentence) for sentence in document]) for document in self.documents]

    def get_sentence_lengths(self):
        return [len(sentence) for document in self.documents for sentence in document]

    def _get_main_tag(self, tag):
        return tag.replace("B-", "").replace("I-", "").replace("O-", "")

    def get_raw_classes(self):
        classes = set()
        for document in self.documents:
            for sentence in document:
                for _, tag in sentence:
                    classes.add(tag)
        return sorted(list(classes))

    def get_classes(self):
        classes = {self._get_main_tag(tag) for tag in self.get_raw_classes()}
        return sorted(list(classes))

    def get_raw_class_sizes(self):
        counter = Counter()
        for document in self.documents:
            for sentence in document:
                for _, tag in sentence:
                    counter.update([tag])
        return counter

    def get_class_sizes(self):
        counter = Counter()
        for document in self.documents:
            for sentence in document:
                for _, tag in sentence:
                    if "-" not in tag or tag.startswith("B-"):
                        counter.update([self._get_main_tag(tag)])
        return counter

    def get_vocabulary_size(self):
        vocabulary = set()
        for document in self.documents:
            for sentence in document:
                for token, _ in sentence:
                    vocabulary.add(token)
        return len(vocabulary)

    def get_unigram_distribution(self):
        class_counts = sorted(self.get_raw_class_sizes().items())
        classes = [tag for tag, _ in class_counts]
        counts = np.array([count for tag, count in class_counts])
        frequencies = counts / sum(counts)
        return classes, frequencies

    def print_unigram_distribution(self):
        classes, frequencies = self.get_unigram_distribution()
        print("\t".join(classes))
        print("\t".join(str(f) for f in frequencies))

    def get_bigram_distribution(self):
        counter = defaultdict(int)
        for document in self.documents:
            for sentence in document:
                for i, (_, tag) in enumerate(sentence):
                    if i > 0:
                        prev_tag = sentence[i-1][1]
                        counter[(prev_tag, tag)] += 1
        classes = self.get_raw_classes()
        counts = []
        bigrams = []
        for class1 in classes:
            for class2 in classes:
                if any([
                    class1 == "O" and class2.startswith("I-"),
                    class1 != class2 and class1.startswith("I-") and class2.startswith("I-"),
                    self._get_main_tag(class1) != self._get_main_tag(class2) and class1.startswith("B-") and class2.startswith("I-"),
                ]):
                    continue  # Invalid bigram
                counts.append(counter[(class1, class2)])
                bigrams.append("{}+{}".format(class1, class2))
        counts = np.array(counts)
        frequencies = counts / sum(counts)
        return bigrams, frequencies

    def print_bigram_distribution(self):
        bigrams, frequencies = self.get_bigram_distribution()
        print("\t".join(bigrams))
        print("\t".join(str(f) for f in frequencies))

    def print_all_distributions(self):
        unigrams, unigram_frequencies = self.get_unigram_distribution()
        bigrams, bigram_frequencies = self.get_bigram_distribution()
        print("\t".join(unigrams + bigrams))
        print("\t".join(str(f) for f in list(unigram_frequencies) + list(bigram_frequencies)))



def print_statistics(dataset_path):
    dataset = CoNLL2003Dataset(dataset_path)
    print("Statistics for {}:".format(dataset_path))
    print()
    print("Number of documents\t{}".format(dataset.get_num_documents()))
    print("Number of sentences\t{}".format(dataset.get_num_sentences()))
    print("Number of tokens\t{}".format(dataset.get_num_tokens()))
    print("Vocabulary size\t{}".format(dataset.get_vocabulary_size()))
    print()
    document_lengths = dataset.get_document_lengths()
    print("Average length of documents\t{}".format(np.mean(document_lengths)))
    print("Shortest document\t{}".format(np.min(document_lengths)))
    print("Longest document\t{}".format(np.max(document_lengths)))
    print()
    sentence_lengths = dataset.get_sentence_lengths()
    print("Average length of sentences\t{}".format(np.mean(sentence_lengths)))
    print("Shortest sentence\t{}".format(np.min(sentence_lengths)))
    print("Longest sentence\t{}".format(np.max(sentence_lengths)))
    print("Number of classes\t{}".format(len(dataset.get_classes())))
    print()
    print("Class distribution:")
    total = 0
    for tag, count in dataset.get_class_sizes().most_common():
        print("{}\t{}".format(tag, count))
        if tag != "O":
            total += count
    print(total)
    # print()
    # print("Tokens per document:")
    # print("\t".join([str(n) for n in dataset.get_document_token_counts()]))
    # print()

if __name__ == "__main__":
    print_statistics(sys.argv[1])
