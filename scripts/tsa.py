import math
import torch


class TSA:

    def __init__(self, num_classes: int, num_steps: int, tensorboard_writer=None):
        self.current_step = 0
        self.num_steps = num_steps
        self.num_classes = num_classes
        self.tensorboard_writer = tensorboard_writer
        self.eta = None

    def step(self):
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('tsa_eta', self.eta)
        self.current_step += 1

    def apply(self, logits, labels, loss_mask):
        probs = torch.softmax(logits.detach(), dim=-1)
        correct_mask = torch.zeros_like(probs).long()
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                correct_mask[i, j, labels[i, j].item()] = 1
        correct_probs = probs[correct_mask == 1].view(labels.shape)
        per_sentence_confidence = correct_probs.mean(-1)
        inconfident_samples = per_sentence_confidence < self.eta
        logits = logits[inconfident_samples]
        labels = labels[inconfident_samples]
        loss_mask = loss_mask[inconfident_samples]

        if self.tensorboard_writer is not None:
            self.tensorboard_writer.add_scalar('tsa_z', sum(inconfident_samples))

        return logits, labels, loss_mask


class LogTSA(TSA):

    def step(self):
        self.eta = (1 - math.exp(-self.current_step/self.num_steps * 5)) * (1 - 1/self.num_classes) + 1/self.num_classes
        super().step()


class LinearTSA(TSA):

    def step(self):
        self.eta = self.current_step/self.num_steps * (1 - 1/self.num_classes) + 1/self.num_classes
        super().step()


class ExpTSA(TSA):

    def step(self):
        self.eta = math.exp((self.current_step/self.num_steps - 1) * 5) * (1 - 1/self.num_classes) + 1/self.num_classes
        super().step()


class ConstantTSA(TSA):

    def step(self):
        self.eta = 1 / self.num_classes
        super().step()
