# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Run BERT on CoNLL-2003 NER with language-adversarial learning."""

from __future__ import absolute_import, division, print_function

import argparse
import collections
import itertools
import logging
from copy import deepcopy

import math
import os
import random
import sys
from io import open

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import (DataLoader, RandomSampler,
                              TensorDataset, SequentialSampler)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from tensorboardX import SummaryWriter

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from pytorch_pretrained_bert.modeling import BertConfig, WEIGHTS_NAME, CONFIG_NAME, BertForTokenClassification
from pytorch_pretrained_bert.optimization import BertAdam, warmup_linear
from pytorch_pretrained_bert.tokenization import BertTokenizer

from conlleval import evaluate

from adversarial import BertForAdversarialFinetuning

if sys.version_info[0] == 2:
    import cPickle as pickle
else:
    import pickle

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


class AdversarialBertForNER(BertForAdversarialFinetuning):

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, loss_mask=None, labels=None, languages=None):
        sequence_output, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=False)
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = loss_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

            adversarial_loss = 0
            adversarial_accuracy = 0
            if languages is not None:
                adversarial_loss, adversarial_accuracy = self.adversarial_loss(sequence_output, languages)
                loss += adversarial_loss

            return loss, adversarial_loss, adversarial_accuracy
        else:
            return logits


class LabelVocab(object):

    def __init__(self):
        self.label_set = set()
        self.labels = []
        self.labels_dict = {}

    def update(self, example):
        self.label_set.update(example.labels)

    def build(self):
        self.labels = ["O"] + sorted(list(self.label_set - {"O"}))
        self.labels_dict = {v: k for k, v in dict(enumerate(self.labels)).items()}

    def convert_labels_to_ids(self, labels):
        return [self.labels_dict[label] for label in labels]

    def convert_ids_to_labels(self, label_ids):
        return [self.labels[label_id] for label_id in label_ids]

    def __len__(self):
        return len(self.label_set)

    def __str__(self):
        return str(self.labels)


class NERExample(object):
    """
    A single training/test example for CoNLL-2003 NER dataset.
    """

    def __init__(self,
                 tokens,
                 labels,
                 language,
                 label_vocab):
        self.tokens = tokens
        self.labels = labels
        self.language = language
        self.label_vocab = label_vocab

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "tokens: [%s]" % (" ".join(self.tokens))
        s += ", labels: [%s]" % (" ".join(self.labels))
        s += ", language: [%s]" % self.language
        return s


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 unique_id,
                 example_index,
                 tokens,
                 token_to_orig_map,
                 input_ids,
                 input_mask,
                 loss_mask,
                 segment_ids,
                 label_ids,
                 language_id,
                 ):
        self.unique_id = unique_id
        self.example_index = example_index
        self.tokens = tokens
        self.token_to_orig_map = token_to_orig_map
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.loss_mask = loss_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.language_id = language_id


def _is_divider(line: str) -> bool:
    empty_line = line.strip() == ''
    if empty_line:
        return True
    else:
        first_token = line.split()[0]
        if first_token == "-DOCSTART-":  # pylint: disable=simplifiable-if-statement
            return True
        else:
            return False


def read_ner_examples(input_file, is_training, languages):
    """Read a CoNLL-2003 file into a list of NERExample."""
    label_vocab = LabelVocab()

    examples = []
    for language in languages:
        language_file = input_file.replace(".lang", ".{}".format(language))
        with open(language_file, "r", encoding='utf-8') as f:
            for is_divider, lines in itertools.groupby(f, _is_divider):
                # Ignore the divider chunks, so that `lines` corresponds to the words
                # of a single sentence.
                if not is_divider:
                    fields = [line.strip().split() for line in lines]
                    # unzipping trick returns tuples, but our Fields need lists
                    fields = [list(field) for field in zip(*fields)]
                    tokens = fields[0]
                    ner_tags = fields[-1]
                    example = NERExample(
                        tokens=tokens,
                        labels=ner_tags,
                        label_vocab=label_vocab,
                        language=language,
                    )
                    label_vocab.update(example)
                    examples.append(example)
    return examples


def convert_examples_to_features(examples, tokenizer, max_seq_length, is_training, languages):
    """Loads a data file into a list of `InputBatch`s."""
    label_vocab = examples[0].label_vocab
    label_vocab.build()
    logger.info("Labels: {}".format(label_vocab))

    unique_id = 1000000000

    features = []

    for (example_index, example) in enumerate(examples):

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_tokens = []
        all_labels = []
        for i, (token, label) in enumerate(zip(example.tokens, example.labels)):
            orig_to_tok_index.append(len(all_tokens))
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_tokens.append(sub_token)
                all_labels.append(label)

        # The -3 accounts for [CLS] and [SEP]
        max_tokens = max_seq_length - 2
        all_tokens = all_tokens[:max_tokens]
        all_labels = all_labels[:max_tokens]

        tokens = []
        segment_ids = []
        labels = []
        tokens.append("[CLS]")
        segment_ids.append(0)
        labels.append('O')
        for token, label in zip(all_tokens, all_labels):
            tokens.append(token)
            segment_ids.append(0)
            labels.append(label)

        tokens.append("[SEP]")
        segment_ids.append(0)
        labels.append('O')

        token_to_orig_map = {(token_index + 1): orig_index for token_index, orig_index in enumerate(tok_to_orig_index)}

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        label_ids = label_vocab.convert_labels_to_ids(labels)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # When computing loss, ignore tail WordPieces
        loss_mask = deepcopy(input_mask)
        for i, token in enumerate(tokens):
            if token.startswith("##"):
                loss_mask[i] = 0

        # Zero-pad up to the sequence length.
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            loss_mask.append(0)
            segment_ids.append(0)
            label_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(loss_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        language_id = languages.index(example.language)

        if example_index < 20:
            logger.info("*** Example ***")
            logger.info("unique_id: %s" % (unique_id))
            logger.info("example_index: %s" % (example_index))
            logger.info("tokens: %s" % " ".join(tokens))
            logger.info("token_to_orig_map: %s" % " ".join([
                "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info(
                "input_mask: %s" % " ".join([str(x) for x in input_mask]))
            logger.info(
                "loss_mask: %s" % " ".join([str(x) for x in loss_mask]))
            logger.info(
                "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
            logger.info("labels: %s" % " ".join(labels))
            logger.info(
                "label_ids: %s" % " ".join([str(x) for x in label_ids]))
            logger.info("language_id: %s" % language_id)

        features.append(
            InputFeatures(
                unique_id=unique_id,
                example_index=example_index,
                tokens=tokens,
                token_to_orig_map=token_to_orig_map,
                input_ids=input_ids,
                input_mask=input_mask,
                loss_mask=loss_mask,
                segment_ids=segment_ids,
                label_ids=label_ids,
                language_id=language_id,
            )
        )
        unique_id += 1

    return features


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "logits"])


def write_predictions(all_examples, all_features, all_results,
                      output_prediction_file, verbose_logging):
    logger.info("Writing predictions to: %s" % (output_prediction_file))

    all_predictions = []
    all_true_labels = []
    for example, features, result in zip(all_examples, all_features, all_results):
        max_scores = result.logits.argmax(dim=1).tolist()
        raw_labels = example.label_vocab.convert_ids_to_labels(max_scores)
        # Remove padding and copy head tags to tail (with B=>I)
        predicted_labels = []
        true_labels = []
        last_orig_index = -1
        for i, token in enumerate(features.tokens):
            if token in ["[CLS]", "[SEP]"]:
                continue
            orig_index = features.token_to_orig_map[i]
            if orig_index == last_orig_index:
                continue  # Tail WordPiece
            # Head WordPiece
            predicted_labels.append(raw_labels[i])
            true_labels.append(example.labels[orig_index])
            last_orig_index = orig_index
        try:
            assert len(predicted_labels) == len(example.labels)
        except AssertionError:
            logger.warning("The following example exceeds the maximum sequence length:\n{}".format(example))

        assert len(true_labels) == len(predicted_labels)
        all_predictions.append(predicted_labels)
        all_true_labels.append(true_labels)

    assert len(all_predictions) == len(all_examples)
    assert len(all_true_labels) == len(all_predictions)

    with open(output_prediction_file, "w") as writer:
        writer.write("-DOCSTART- -X- -X- O" + "\n\n")
        for labels in all_predictions:
            writer.write("\n".join(labels) + "\n\n")

    flat_true_labels = list(itertools.chain(*all_true_labels))
    flat_predicted_labels = list(itertools.chain(*all_predictions))
    assert len(flat_true_labels) == len(flat_predicted_labels)
    return evaluate(flat_true_labels, flat_predicted_labels, verbose=True)


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--bert_model", default=None, type=str, required=True,
                        help="Bert pre-trained model selected in the list: bert-base-uncased, "
                             "bert-large-uncased, bert-base-cased, bert-large-cased, bert-base-multilingual-uncased, "
                             "bert-base-multilingual-cased, bert-base-chinese.")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model checkpoints and predictions will be written.")

    ## Other parameters
    parser.add_argument("--train_file", default=None, type=str)
    parser.add_argument("--predict_file", default=None, type=str)
    parser.add_argument("--max_seq_length", default=384, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences "
                             "longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_predict", action='store_true', help="Whether to run eval on the dev set.")
    parser.add_argument("--train_batch_size", default=32, type=int, help="Total batch size for training.")
    parser.add_argument("--predict_batch_size", default=8, type=int, help="Total batch size for predictions.")
    parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=3.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% "
                             "of training.")
    parser.add_argument("--verbose_logging", action='store_true',
                        help="If true, all of the warnings related to data processing will be printed.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--do_lower_case",
                        action='store_true',
                        help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--fp16',
                        action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--loss_scale',
                        type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    parser.add_argument('--evaluate_each_epoch',
                        action='store_true',
                        help="Whether to run the evaluation script after every training epoch")
    parser.add_argument('--early_stopping',
                        action='store_true',
                        help="Whether to stop finetuning of F1 score on validation set does not improve")
    parser.add_argument('--train_languages', nargs='+', help='<Required> Finetuning languages', required=False)
    parser.add_argument('--predict_languages', nargs='+', help='Validation/prediction languages', required=False)
    args = parser.parse_args()

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl')
    logger.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
        device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if not args.do_train and not args.do_predict:
        raise ValueError("At least one of `do_train` or `do_predict` must be True.")

    if args.do_train:
        if not args.train_file:
            raise ValueError(
                "If `do_train` is True, then `train_file` must be specified.")
    if args.do_predict:
        if not args.predict_file:
            raise ValueError(
                "If `do_predict` is True, then `predict_file` must be specified.")

    if os.path.exists(args.output_dir) and len(os.listdir(args.output_dir)) > 1 and args.do_train:
        raise ValueError("Output directory () already exists and is not empty.")
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    tensorboard_writer = SummaryWriter(os.path.join(args.output_dir, "runs"))

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)

    train_examples = None
    num_train_optimization_steps = None
    if args.do_train:
        train_examples = read_ner_examples(
            input_file=args.train_file, is_training=True, languages=args.train_languages)
        num_train_optimization_steps = int(
            len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
        if args.local_rank != -1:
            num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()

    # Prepare model
    model = AdversarialBertForNER.from_pretrained(args.bert_model,
                                                  cache_dir=os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE),
                                                                         'distributed_{}'.format(args.local_rank)),
                                                  num_labels=len(
                                                      train_examples[0].label_vocab) if train_examples else 1,
                                                  num_languages=len(args.train_languages or args.predict_languages))

    if args.fp16:
        model.half()
    model.to(device)
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())

    # hack to remove pooler, which is not used
    # thus it produce None grad that break apex
    param_optimizer = [n for n in param_optimizer if 'pooler' not in n[0]]

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    if args.fp16:
        try:
            from apex.optimizers import FP16_Optimizer
            from apex.optimizers import FusedAdam
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        optimizer = FusedAdam(optimizer_grouped_parameters,
                              lr=args.learning_rate,
                              bias_correction=False,
                              max_grad_norm=1.0)
        if args.loss_scale == 0:
            optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
        else:
            optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
    else:
        optimizer = BertAdam(optimizer_grouped_parameters,
                             lr=args.learning_rate,
                             warmup=args.warmup_proportion,
                             t_total=num_train_optimization_steps)

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        eval_examples = read_ner_examples(
            input_file=args.predict_file, is_training=False, languages=args.predict_languages)
        eval_features = convert_examples_to_features(
            examples=eval_examples,
            tokenizer=tokenizer,
            max_seq_length=args.max_seq_length,
            is_training=False, languages=args.predict_languages)

        input_filename = os.path.basename(args.predict_file).replace(".lang", "." + "_".join(args.predict_languages))
        output_filepath = os.path.join(args.output_dir, input_filename + ".predictions.txt")

        def evaluate_model(model):
            logger.info("***** Running predictions *****")
            logger.info("  Num orig examples = %d", len(eval_examples))
            logger.info("  Num split examples = %d", len(eval_features))
            logger.info("  Batch size = %d", args.predict_batch_size)
            all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
            all_loss_mask = torch.tensor([f.loss_mask for f in eval_features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
            all_example_index = torch.arange(all_input_ids.size(0), dtype=torch.long)
            eval_data = TensorDataset(all_input_ids, all_input_mask, all_loss_mask, all_segment_ids, all_example_index)
            # Run prediction for full data
            eval_sampler = SequentialSampler(eval_data)
            eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.predict_batch_size)
            model.eval()
            all_results = []
            logger.info("Start evaluating")
            for input_ids, input_mask, loss_mask, segment_ids, example_indices in tqdm(eval_dataloader,
                                                                                       desc="Evaluating"):
                if len(all_results) % 1000 == 0:
                    logger.info("Processing example: %d" % (len(all_results)))
                input_ids = input_ids.to(device)
                input_mask = input_mask.to(device)
                loss_mask = loss_mask.to(device)
                segment_ids = segment_ids.to(device)
                with torch.no_grad():
                    batch_logits = model(input_ids, segment_ids, input_mask, loss_mask)
                for i, example_index in enumerate(example_indices):
                    logits = batch_logits[i].detach().cpu()
                    eval_feature = eval_features[example_index.item()]
                    unique_id = int(eval_feature.unique_id)
                    all_results.append(RawResult(unique_id=unique_id,
                                                 logits=logits))
            return write_predictions(eval_examples, eval_features, all_results, output_filepath,
                                     args.verbose_logging)
    else:
        def evaluate_model(model): pass

    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
    output_model_file = os.path.join(args.output_dir, WEIGHTS_NAME)

    global_step = 0
    if args.do_train:
        cached_train_features_file = args.train_file + '_{0}_{1}'.format(
            list(filter(None, args.bert_model.split('/'))).pop(), str(args.max_seq_length))
        train_features = None
        try:
            with open(cached_train_features_file, "rb") as reader:
                train_features = pickle.load(reader)
        except:
            train_features = convert_examples_to_features(
                examples=train_examples,
                tokenizer=tokenizer,
                max_seq_length=args.max_seq_length,
                is_training=True,
                languages=args.train_languages)
            if args.local_rank == -1 or torch.distributed.get_rank() == 0:
                logger.info("  Saving train features into cached file %s", cached_train_features_file)
                with open(cached_train_features_file, "wb") as writer:
                    pickle.dump(train_features, writer)
        logger.info("***** Running training *****")
        logger.info("  Num orig examples = %d", len(train_examples))
        logger.info("  Num split examples = %d", len(train_features))
        logger.info("  Num labels = %d", len(train_examples[0].label_vocab))
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Num steps = %d", num_train_optimization_steps)
        all_input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long)
        all_loss_mask = torch.tensor([f.loss_mask for f in train_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long)
        all_labels = torch.tensor([f.label_ids for f in train_features], dtype=torch.long)
        all_language_ids = torch.tensor([f.language_id for f in train_features], dtype=torch.long)
        train_data = TensorDataset(all_input_ids, all_input_mask, all_loss_mask, all_segment_ids, all_labels,
                                   all_language_ids)
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

        current_f1 = 0.0
        best_f1 = 0.0

        for epoch in trange(int(args.num_train_epochs), desc="Epoch"):
            model.train()
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
                if n_gpu == 1:
                    batch = tuple(t.to(device) for t in batch)  # multi-gpu does scattering it-self
                input_ids, input_mask, loss_mask, segment_ids, labels, language_ids = batch
                loss, adversarial_loss, adversarial_accuracy = model(input_ids, segment_ids, input_mask, loss_mask, labels, language_ids)
                if n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps

                if args.fp16:
                    optimizer.backward(loss)
                else:
                    loss.backward()
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        # modify learning rate with special warm up BERT uses
                        # if args.fp16 is False, BertAdam is used and handles this automatically
                        lr_this_step = args.learning_rate * warmup_linear(global_step / num_train_optimization_steps,
                                                                          args.warmup_proportion)
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr_this_step

                    tensorboard_writer.add_scalar('loss', loss.item())
                    tensorboard_writer.add_scalar('adversarial_loss', adversarial_loss.item())
                    tensorboard_writer.add_scalar('adversarial_accuracy', adversarial_accuracy)
                    optimizer_params = optimizer.param_groups[-1]
                    tensorboard_writer.add_scalar('weight_decay', optimizer_params["weight_decay"])
                    tensorboard_writer.add_scalar('learning_rate', optimizer_params["lr"])

                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

            if args.evaluate_each_epoch:
                precision, recall, f1 = evaluate_model(model)
                tensorboard_writer.add_scalar('precision', precision)
                tensorboard_writer.add_scalar('recall', recall)
                tensorboard_writer.add_scalar('f1', f1)
                if args.early_stopping and epoch > 0:
                    if f1 < current_f1:
                        logger.info("Stopping early because {} F1 < {} F1".format(f1, current_f1))
                        break

                if f1 > best_f1:
                    logger.info("Saving model ...")
                    model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                    torch.save(model_to_save.state_dict(), output_model_file)
                    output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                    with open(output_config_file, 'w') as f:
                        f.write(model_to_save.config.to_json_string())
                    best_f1 = f1

                current_f1 = f1

            if not args.evaluate_each_epoch:
                logger.info("Saving model ...")
                model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
                torch.save(model_to_save.state_dict(), output_model_file)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                with open(output_config_file, 'w') as f:
                    f.write(model_to_save.config.to_json_string())

    del model

    if args.do_predict and (args.local_rank == -1 or torch.distributed.get_rank() == 0):
        # Load a trained model and config that you have fine-tuned
        config = BertConfig(output_config_file)
        model = AdversarialBertForNER(config, num_labels=len(eval_examples[0].label_vocab), num_languages=2)
        model.load_state_dict(torch.load(output_model_file))
        model.to(device)
        evaluate_model(model)


if __name__ == "__main__":
    main()
