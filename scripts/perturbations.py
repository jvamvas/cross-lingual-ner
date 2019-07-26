import random
from copy import deepcopy
from string import ascii_lowercase, ascii_uppercase

import torch
from pytorch_pretrained_bert import BertTokenizer, BertForMaskedLM


class Perturbation:

    def __init__(self, device):
        self.device = device


class CharReplacePerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float, names_only=False):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.vocab_set = set(self.tokenizer.vocab.keys())
        self.token_rate = token_rate
        self.names_only = names_only

    def perturbe(self, batch, logits):
        perturbed_batch = deepcopy(batch)
        input_ids, input_mask, loss_mask, segment_ids = perturbed_batch
        perturbed_token_mask = torch.rand(input_ids.shape) < self.token_rate
        perturbed_token_mask = perturbed_token_mask.to(self.device)
        perturbed_token_mask *= input_mask.byte()  # Ignore pads
        if self.names_only:
            names_mask = (logits.argmax(dim=-1) > 0).long()  # 0 is ID of "O" tag
            perturbed_token_mask *= names_mask.byte()
        for perturbed_token_index in perturbed_token_mask.nonzero():
            original_id = input_ids[list(perturbed_token_index)].item()
            token = self.tokenizer.convert_ids_to_tokens([original_id])[0]
            if token in {"[CLS]", "[SEP]", "[PAD]", "[MASK]"}:
                continue
            if len(token) == 1:
                continue  # Do not want to replace first character of a word

            if token.startswith("##"):
                start_index = 2
            else:
                start_index = 0
            replaced_indices = list(range(start_index, len(token) - 1))
            random.shuffle(replaced_indices)
            perturbed_token = None
            for replaced_index in replaced_indices:
                isupper = token[replaced_index].isupper()
                charbase = ascii_uppercase if isupper else ascii_lowercase
                replacement_candidates = list(charbase.replace(token[replaced_index], ''))
                random.shuffle(replacement_candidates)
                for replacement in replacement_candidates:
                    candidate_token = token[:replaced_index] + replacement + token[replaced_index+1:]
                    if candidate_token in self.vocab_set:
                        perturbed_token = candidate_token
                        break
                if perturbed_token is not None:
                    break
            if perturbed_token is None:
                perturbed_token = token
            replaced_id = self.tokenizer.convert_tokens_to_ids([perturbed_token])[0]
            input_ids[list(perturbed_token_index)] = replaced_id
        return perturbed_batch


class CharRemovePerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float, names_only=False):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.vocab_set = set(self.tokenizer.vocab.keys())
        self.token_rate = token_rate
        self.names_only = names_only

    def perturbe(self, batch, logits):
        perturbed_batch = deepcopy(batch)
        input_ids, input_mask, loss_mask, segment_ids = perturbed_batch
        perturbed_token_mask = torch.rand(input_ids.shape) < self.token_rate
        perturbed_token_mask = perturbed_token_mask.to(self.device)
        perturbed_token_mask *= input_mask.byte()  # Ignore pads
        if self.names_only:
            names_mask = (logits.argmax(dim=-1) > 0).byte()  # 0 is ID of "O" tag
            perturbed_token_mask *= names_mask
        for perturbed_token_index in perturbed_token_mask.nonzero():
            original_id = input_ids[list(perturbed_token_index)].item()
            token = self.tokenizer.convert_ids_to_tokens([original_id])[0]
            if token in {"[CLS]", "[SEP]", "[PAD]", "[MASK]"}:
                continue
            if len(token) <= 2:
                continue

            if token.startswith("##"):
                start_index = 2
            else:
                start_index = 0
            replace_indices = list(range(start_index, len(token) - 1))
            random.shuffle(replace_indices)
            perturbed_token = None
            for replaced_index in replace_indices:
                candidate_token = token[:replaced_index] + token[replaced_index+1:]
                if replaced_index == 0 and token[0].isupper():
                    candidate_token = token[1].upper() + token[2:]
                if candidate_token in self.vocab_set:
                    perturbed_token = candidate_token
                    break
            if perturbed_token is not None:
                replaced_id = self.tokenizer.convert_tokens_to_ids([perturbed_token])[0]
                input_ids[list(perturbed_token_index)] = replaced_id
        return perturbed_batch


class DropTailPerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float, names_only=False):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.vocab_set = set(self.tokenizer.vocab.keys())
        self.token_rate = token_rate
        self.default_tail_id = None
        for id, token in tokenizer.ids_to_tokens.items():
            if token.startswith("##"):
                self.default_tail_id = id
                break
        assert self.default_tail_id is not None
        self.names_only = names_only

    def perturbe(self, batch, logits):
        perturbed_batch = deepcopy(batch)
        input_ids, input_mask, loss_mask, segment_ids = perturbed_batch
        perturbed_token_mask = torch.rand(input_ids.shape) < self.token_rate
        perturbed_token_mask = perturbed_token_mask.to(self.device)
        perturbed_token_mask *= input_mask.byte()  # Ignore pads
        if self.names_only:
            names_mask = (logits.argmax(dim=-1) > 0).long()  # 0 is ID of "O" tag
            perturbed_token_mask *= names_mask
        for perturbed_token_index in perturbed_token_mask.nonzero():
            original_id = input_ids[list(perturbed_token_index)].item()
            token = self.tokenizer.convert_ids_to_tokens([original_id])[0]
            if not token.startswith("##"):
                continue
            else:
                input_ids[list(perturbed_token_index)] = self.default_tail_id
        return perturbed_batch


class BothPerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float):
        super().__init__(device)
        self.char_replace = CharReplacePerturbation(device, tokenizer, token_rate)
        self.drop_tail = DropTailPerturbation(device, tokenizer, token_rate)

    def perturbe(self, batch, logits):
        perturbed_batch = self.char_replace.perturbe(batch, logits)
        perturbed_batch = self.drop_tail.perturbe(perturbed_batch, logits)
        return perturbed_batch


class CasePerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.vocab_set = set(self.tokenizer.vocab.keys())
        self.token_rate = token_rate

    def perturbe(self, batch, logits):
        perturbed_batch = deepcopy(batch)
        input_ids, input_mask, loss_mask, segment_ids = perturbed_batch
        perturbed_token_mask = torch.rand(input_ids.shape) < self.token_rate
        perturbed_token_mask = perturbed_token_mask.to(self.device)
        perturbed_token_mask *= input_mask.byte()  # Ignore pads
        for perturbed_token_index in perturbed_token_mask.nonzero():
            original_id = input_ids[list(perturbed_token_index)].item()
            token = self.tokenizer.convert_ids_to_tokens([original_id])[0]
            if token.startswith("##"):
                continue
            if len(token) == 1:
                continue  # e.g. avoid doesn'T
            perturbed_token = token[0].swapcase() + token[1:]
            if perturbed_token not in self.vocab_set:
                continue
            replaced_id = self.tokenizer.convert_tokens_to_ids([perturbed_token])[0]
            input_ids[list(perturbed_token_index)] = replaced_id
        return perturbed_batch


class MaskPerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float, exclude_names: bool = False):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.token_rate = token_rate
        self.exclude_names = exclude_names
        self.mask_id = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]
        self.sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

    def perturbe(self, batch, logits):
        input_ids, input_mask, loss_mask, segment_ids = batch
        perturbed_token_mask = (torch.rand(input_ids.shape) < self.token_rate).long().to(self.device)
        perturbed_token_mask *= input_mask  # Ignore pads
        if self.exclude_names:
            nonames_mask = (logits.argmax(dim=-1) == 0).long()  # 0 is ID of "O" tag
            perturbed_token_mask *= nonames_mask
        cls_mask = torch.ones_like(input_ids)
        cls_mask[:, 0] = 0
        sep_mask = (input_ids != self.sep_id).long()
        perturbed_token_mask *= cls_mask * sep_mask
        input_ids = input_ids * (perturbed_token_mask ^ 1) + self.mask_id * perturbed_token_mask
        return input_ids, input_mask, loss_mask, segment_ids


class WordReplacePerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float, exclude_names: bool = False):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.token_rate = token_rate
        self.exclude_names = exclude_names
        self.sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]

    def perturbe(self, batch, logits):
        input_ids, input_mask, loss_mask, segment_ids = batch
        perturbed_token_mask = (torch.rand(input_ids.shape) < self.token_rate).long().to(self.device)
        perturbed_token_mask *= input_mask  # Ignore pads
        if self.exclude_names:
            nonames_mask = (logits.argmax(dim=-1) == 0).long()  # 0 is ID of "O" tag
            perturbed_token_mask *= nonames_mask
        cls_mask = torch.ones_like(input_ids)
        cls_mask[:, 0] = 0
        sep_mask = (input_ids != self.sep_id).long()
        perturbed_token_mask *= cls_mask * sep_mask
        id_pool = []
        for perturbed_token_index in perturbed_token_mask.nonzero():
            original_id = input_ids[list(perturbed_token_index)].item()
            id_pool.append(original_id)
        random.shuffle(id_pool)
        for perturbed_token_index, replacement_id in zip(perturbed_token_mask.nonzero(), id_pool):
            input_ids[list(perturbed_token_index)] = replacement_id
        return input_ids, input_mask, loss_mask, segment_ids


class SwapPerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float, exclude_names: bool = False):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.token_rate = token_rate
        self.exclude_names = exclude_names
        self.mask_id = tokenizer.convert_tokens_to_ids(["[MASK]"])[0]

    def perturbe(self, batch, logits):
        perturbed_batch = deepcopy(batch)
        input_ids, input_mask, loss_mask, segment_ids = perturbed_batch
        perturbed_token_mask = torch.rand(input_ids.shape) < self.token_rate
        perturbed_token_mask = perturbed_token_mask.to(self.device)
        perturbed_token_mask *= input_mask.byte()  # Ignore pads
        left_mask = torch.ones_like(input_ids)
        left_mask[:, :2] = 0  # Ensure that there is a left neighbour that is not [CLS]
        perturbed_token_mask *= left_mask.byte()
        if self.exclude_names:
            nonames_mask_b = logits.argmax(dim=-1) == 0  # 0 is ID of "O" tag
            perturbed_token_mask *= nonames_mask_b
            nonames_mask_a = torch.cat((nonames_mask_b[:, 1:], torch.ones((nonames_mask_b.shape[0], 1)).byte().to(self.device)), dim=-1)
            perturbed_token_mask *= nonames_mask_a
        for index_b in perturbed_token_mask.nonzero():
            index_a = index_b + torch.LongTensor([0, -1]).to(self.device)
            index_c = index_b + torch.LongTensor([0, 1]).to(self.device)  # Token right to the swapped pair
            id_a = input_ids[list(index_a)].item()
            id_b = input_ids[list(index_b)].item()
            id_c = input_ids[list(index_c)].item()
            tokens = self.tokenizer.convert_ids_to_tokens([id_a, id_b, id_c])
            if any([token.startswith("##") for token in tokens]):
                break  # Do not change multi-token words
            if any([len(token) < 2 for token in tokens[:2]]):
                break  # E.g. punctuation
            input_ids[list(index_a)] = id_b
            input_ids[list(index_b)] = id_a
        return perturbed_batch


class MaskReplacePerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float):
        super().__init__(device)
        self.char_replace = CharReplacePerturbation(device, tokenizer, token_rate=1, names_only=True)
        self.mask = MaskPerturbation(device, tokenizer, token_rate, exclude_names=True)

    def perturbe(self, batch, logits):
        perturbed_batch = self.char_replace.perturbe(batch, logits)
        perturbed_batch = self.mask.perturbe(perturbed_batch, logits)
        return perturbed_batch


class MaskWordReplacePerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float, exclude_names):
        super().__init__(device)
        self.mask = MaskPerturbation(device, tokenizer, token_rate=(0.8 * token_rate), exclude_names=exclude_names)
        self.word_replace = WordReplacePerturbation(device, tokenizer, token_rate=(0.2 * token_rate), exclude_names=True)

    def perturbe(self, batch, logits):
        perturbed_batch = self.mask.perturbe(batch, logits)
        perturbed_batch = self.word_replace.perturbe(perturbed_batch, logits)
        return perturbed_batch


class MaskRemovePerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float):
        super().__init__(device)
        self.char_remove = CharRemovePerturbation(device, tokenizer, token_rate, names_only=True)
        self.mask = MaskPerturbation(device, tokenizer, token_rate, exclude_names=True)

    def perturbe(self, batch, logits):
        perturbed_batch = self.char_remove.perturbe(batch, logits)
        perturbed_batch = self.mask.perturbe(perturbed_batch, logits)
        return perturbed_batch


class FilledPerturbation(Perturbation):

    def __init__(self, device, tokenizer: BertTokenizer, token_rate: float, exclude_names: bool = False):
        super().__init__(device)
        self.tokenizer = tokenizer
        self.token_rate = token_rate
        self.exclude_names = exclude_names
        self.mask = MaskPerturbation(device, tokenizer, token_rate)
        self.model = BertForMaskedLM.from_pretrained("bert-base-multilingual-cased")
        self.model.eval()
        self.model.to(device)

    def perturbe(self, batch, logits):
        masked_batch = self.mask.perturbe(batch, logits)
        input_ids, input_mask, loss_mask, segment_ids = masked_batch
        with torch.no_grad():
            predictions = self.model(input_ids)
        predicted_ids = torch.argmax(predictions, dim=-1)
        masked_mask = (input_ids == self.mask.mask_id).long()
        input_ids = input_ids * (masked_mask ^ 1) + predicted_ids * masked_mask
        return input_ids, input_mask, loss_mask, segment_ids


def load_perturbation_from_descriptor(descriptor: str, device, tokenizer: BertTokenizer):
    exclude_names = "_noname" in descriptor
    if descriptor.startswith("char_replace"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = CharReplacePerturbation(device, tokenizer, token_rate)
    elif descriptor.startswith("char_remove"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = CharRemovePerturbation(device, tokenizer, token_rate, names_only=True)
    elif descriptor.startswith("drop_tail"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = DropTailPerturbation(device, tokenizer, token_rate)
    elif descriptor.startswith("both"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = BothPerturbation(device, tokenizer, token_rate)
    elif descriptor.startswith("case"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = CasePerturbation(device, tokenizer, token_rate)
    elif descriptor.startswith("mask_replace"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = MaskReplacePerturbation(device, tokenizer, token_rate)
    elif descriptor.startswith("mask_word_replace"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = MaskWordReplacePerturbation(device, tokenizer, token_rate, exclude_names)
    elif descriptor.startswith("filled"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = FilledPerturbation(device, tokenizer, token_rate)
    elif descriptor.startswith("word_replace"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = WordReplacePerturbation(device, tokenizer, token_rate, exclude_names)
    elif descriptor.startswith("mask_remove"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = MaskRemovePerturbation(device, tokenizer, token_rate)
    elif descriptor.startswith("mask"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = MaskPerturbation(device, tokenizer, token_rate, exclude_names)
    elif descriptor.startswith("swap"):
        token_rate = float(descriptor.split("_")[-1])
        perturbation = SwapPerturbation(device, tokenizer, token_rate, exclude_names)
    else:
        raise NotImplementedError()
    return perturbation
