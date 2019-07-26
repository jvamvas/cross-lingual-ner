# -*- coding: utf-8 -*
import sys
from copy import deepcopy

from pytorch_pretrained_bert import BertTokenizer

from conll_subset import CoNLL2003Dataset


def maximize_coverage(source: CoNLL2003Dataset, target: CoNLL2003Dataset, n: int, tokenizer: BertTokenizer) -> CoNLL2003Dataset:
    MAX_SEQ_LEN = 150

    target_vocab = set()
    for document in target.documents:
        for sentence in document:
            for token, ner_tag in sentence:
                if token is None:
                    continue
                word_pieces = tokenizer.tokenize(token)
                target_vocab.update(word_pieces)

    annotated_train_sentences = []
    for document in source.documents:
        annotated_train_sentences += document

    tokenized_train_sentences = []
    for i, annotated_sentence in enumerate(annotated_train_sentences):
        sentence_word_pieces = []
        for token, _ in annotated_sentence:
            if token is None:
                continue
            word_pieces = tokenizer.tokenize(token)
            sentence_word_pieces += word_pieces
        sentence_word_pieces = set(sentence_word_pieces[:MAX_SEQ_LEN])
        coverage = len(target_vocab & sentence_word_pieces)
        tokenized_train_sentences.append({
            "id": i,
            "set": sentence_word_pieces,
            "coverage": coverage,
        })

    selected_train_sentences = []
    for i in range(n):
        tokenized_train_sentences.sort(key=lambda s: s["coverage"])
        best_sentence = tokenized_train_sentences.pop()
        selected_train_sentences.append(annotated_train_sentences[best_sentence["id"]])
        new_word_pieces = target_vocab & best_sentence["set"]
        for new_word_piece in new_word_pieces:
            target_vocab.remove(new_word_piece)
            for j in range(len(tokenized_train_sentences)):
                if new_word_piece in tokenized_train_sentences[j]["set"]:
                    tokenized_train_sentences[j]["set"].remove(new_word_piece)
                    tokenized_train_sentences[j]["coverage"] -= 1

    output = deepcopy(source)
    output.documents = [[sentence] for sentence in selected_train_sentences]
    return output


if __name__ == "__main__":
    source = CoNLL2003Dataset(sys.argv[1])
    target = CoNLL2003Dataset(sys.argv[2])
    n = int(sys.argv[3])
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
    output = maximize_coverage(source, target, n, tokenizer)
    print(output)
