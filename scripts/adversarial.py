"""
pyTorch submodules that are needed for language-adversarial learning.
"""

import math
import torch
from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from torch import nn
from torch.autograd import Function


class BertForAdversarialFinetuning(BertPreTrainedModel):
    """
    Version of BERT that includes a language discriminator
    """

    def __init__(self, config, num_labels, num_languages):
        super().__init__(config)
        self.num_labels = num_labels
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        self.adversary = TokenLanguageDiscriminator(
            input_size=config.hidden_size,
            hidden_size=math.floor(config.hidden_size / 2),
            num_languages=num_languages,
        )
        self.apply(self.init_bert_weights)
        self.adversarial_weight = 1

    def adversarial_loss(self, sequence_output, languages):
        common_half = sequence_output[:, :, :self.adversary.input_size]
        language_scores = self.adversary(common_half)
        language_loss = self.adversary.compute_loss(language_scores, languages)
        language_accuracy = self.adversary.compute_accuracy(language_scores, languages)
        return self.adversarial_weight * language_loss, language_accuracy


class TokenLanguageDiscriminator(nn.Module):
    """
    Language Discriminator module that classifies each token
    """

    def __init__(self, input_size, hidden_size, num_languages):
        super().__init__()
        self.input_size = input_size
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, num_languages))
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input = grad_reverse(input)

        input_shape = input.shape
        input = input.reshape(input_shape[0] * input_shape[1], -1)
        scores = self.classifier(input)
        scores = scores.reshape(input_shape[0], input_shape[1], -1)
        return scores

    def compute_accuracy(self, scores, language):
        language_preds = scores.argmax(dim=2)
        language_preds = language_preds.reshape(-1)
        if len(language):
            return float(torch.sum(language_preds == language.repeat(scores.shape[1]))) / language_preds.numel()
        else:
            return 0

    def compute_loss(self, scores, language):
        return self.loss(scores.reshape(scores.shape[0] * scores.shape[1], -1),
                         language.repeat(scores.shape[1]))


class GradReverse(Function):
    """
    Gradient reversal layer
    """

    def forward(self, x, **kwargs):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output.neg()


def grad_reverse(x):
    return GradReverse()(x)
