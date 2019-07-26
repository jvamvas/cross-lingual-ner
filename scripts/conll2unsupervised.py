import itertools
import random
import sys
from copy import deepcopy


class CoNLL2003Dataset:

    def __init__(self, filepath):
        self.documents = []
        with open(filepath) as f:
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

    def print_unsupervised(self):
        for document in self.documents:
            for sentence in document:
                line = " ".join([token for token, _ in sentence])
                if line.strip():
                    print(line)

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
    dataset = CoNLL2003Dataset(sys.argv[1])
    dataset.print_unsupervised()
