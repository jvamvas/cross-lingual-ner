# Cross-Lingual NER

Code accompanying the M.Sc. thesis "Cross-Lingual Named Entity Recognition" submitted to LMU Munich.

## Dependencies
The code was tested with the following dependency versions:
* Python 3.6
* pyTorch 1.0.1
* pytorch-pretrained-bert 0.6.1 (newest version is known as PyTorch-Transformers, but the dependency is still available on PyPI as of 26. July 2019)

## Installation
* Create virtual environment
* `pip install -r requirements.txt`
* Place CoNLL-formatted data in the "data" directory

## Main components
### scripts/run_ner.py

Implementation of BERT fine-tuning for NER.

Based on the question-answering script on https://github.com/huggingface/pytorch-transformers/blob/d1e4fa98a91dfebfd88ef77b5ee761665ea0fe4a/examples/run_squad.py.

Example usage:
```bash
python scripts/run_ner.py \
  --bert_model bert-base-cased \
  --do_train \
  --do_predict \
  --train_file data/train.txt \
  --predict_file data/valid.txt \
  --train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 150 \
  --output_dir output/my_model
```

Example usage with expectation regularization:
```bash
python scripts/run_ner.py \
  --bert_model bert-base-cased \
  --do_train \
  --do_predict \
  --train_file data/train.txt \
  --predict_file data/valid.txt \
  --train_batch_size 16 \
  --learning_rate 3e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 150 \
  --expectation_regularization \
  --expectation_regularization_weight 1 \
  --unsupervised_file data/unsupervised.txt \
  --output_dir output/my_model
```

### scripts/run_adversarial_ner.py

Modified script that implements language-adversarial fine-tuning.

Example usage (assuming you have datasets train.en, valid.en, train.de, valid.de in the data directory):
```bash
python scripts/run_adversarial_ner.py \
  --bert_model bert-base-multilingual-cased \
  --do_train \
  --do_predict \
  --train_file data/train.lang \
  --predict_file data/valid.lang \
  --train_batch_size 32 \
  --predict_batch_size 256 \
  --learning_rate 4e-5 \
  --num_train_epochs 4.0 \
  --max_seq_length 150 \
  --evaluate_each_epoch \
  --train_languages en de \
  --predict_languages en de \
  --output_dir output/my_model
```

### scripts/run_uda_ner.py

Modified script that implements unsupervised data augmentation (Xie, Qizhe, et al. 2019: "Unsupervised data augmentation." arXiv preprint arXiv:1904.12848)

```bash
run_uda_ner.py \
  --pretrained_bert_model bert-base-cased \
  --do_train \
  --do_predict \
  --train_file data/train.txt \
  --predict_file data/valid.txt \
  --train_batch_size 15 \
  --unsupervised_batch_size 150 \
  --learning_rate 3e-6 \
  --num_train_epochs 100.0 \
  --max_seq_length 150 \
  --unsupervised_file data/unsupervised.txt \
  --perturbation mask_0.15 \
  --unsupervised_weight 1 \
  --tsa linear_9 \
  --output_dir output/my_model
```

### Miscellaneous Scripts

**conll2unsupervised.py**: Generate an unsupervised corpus from a CoNLL-formatted annotated dataset.

**conll_sampling.py**: Take a sample from a CoNLL-formatted dataset, with a given number of sentences.

**conll_sampling_fixed_tokens.py**: Take a sample from a CoNLL-formatted dataset that has a given number of sentences and approximates a given number of tokens.

**conll_statistics.py**: Print a table with statistics for a given CoNLL dataset.

**maximize_coverage.py**: Sample n sentences from a CoNLL training set such that the sample has (approximately) maximum subwords coverage of a provided validation set.

**measure_coverage.py**: Compute four different subword coverage measures between a two CoNLL datasets.