from unittest import TestCase

from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from scripts.perturbations import *
from scripts.run_uda_ner import read_unsupervised_examples, convert_unsupervised_examples_to_features


class PerturbationsTestCase(TestCase):

    def setUp(self) -> None:
        self.batch_size = 10
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
        self.device = None
        self.unsupervised_filepath = "../tests/data/unsupervised.txt"
        self.unsupervised_examples = read_unsupervised_examples(input_file=self.unsupervised_filepath)
        unsupervised_features = convert_unsupervised_examples_to_features(
            examples=self.unsupervised_examples,
            tokenizer=self.tokenizer,
            max_seq_length=150,
        )
        unsupervised_input_ids = torch.tensor([f.input_ids for f in unsupervised_features], dtype=torch.long)
        unsupervised_input_mask = torch.tensor([f.input_mask for f in unsupervised_features], dtype=torch.long)
        unsupervised_loss_mask = torch.tensor([f.loss_mask for f in unsupervised_features], dtype=torch.long)
        unsupervised_segment_ids = torch.tensor([f.segment_ids for f in unsupervised_features], dtype=torch.long)
        unsupervised_data = TensorDataset(unsupervised_input_ids, unsupervised_input_mask, unsupervised_loss_mask,
                                          unsupervised_segment_ids)
        unsupervised_sampler = SequentialSampler(unsupervised_data)
        self.unsupervised_dataloader = DataLoader(unsupervised_data, sampler=unsupervised_sampler,
                                             batch_size=self.batch_size)

    def _ids_to_text(self, ids):
        tokens = self.tokenizer.convert_ids_to_tokens([int(id) for id in ids])
        return " ".join(tokens).replace("[PAD]", "").strip()

    def _print_sentence(self, batch):
        input_ids, input_mask, loss_mask, segment_ids = batch
        for sentence in input_ids:
            print(self._ids_to_text(sentence))
        print()

    def test_char_replace_perturbation(self):
        self.char_replace_perturbation = CharReplacePerturbation(self.device, self.tokenizer, token_rate=0.5)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        perturbed_batch = self.char_replace_perturbation.perturbe(unsupervised_batch, None)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)

    def test_drop_tail_perturbation(self):
        self.drop_tail_perturbation = DropTailPerturbation(self.device, self.tokenizer, token_rate=0.5)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        perturbed_batch = self.drop_tail_perturbation.perturbe(unsupervised_batch, None)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)

    def test_case_perturbation(self):
        self.case_perturbation = CasePerturbation(self.device, self.tokenizer, token_rate=0.5)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        perturbed_batch = self.case_perturbation.perturbe(unsupervised_batch, None)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)

    def test_mask_perturbation(self):
        self.mask_perturbation = MaskPerturbation(self.device, self.tokenizer, token_rate=0.5)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        perturbed_batch = self.mask_perturbation.perturbe(unsupervised_batch, None)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)

    def test_mask_perturbation_exclude_names(self):
        self.mask_perturbation = load_perturbation_from_descriptor("mask_noname_1", self.device, self.tokenizer)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        input_ids, input_mask, loss_mask, segment_ids = unsupervised_batch
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 9)
        # All tokens are no names
        logits[:, :, 0] = 1
        perturbed_batch = self.mask_perturbation.perturbe(unsupervised_batch, logits)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)
        # First token is no name
        logits[0, 0, 0] = 0.1
        logits[0, 0, 1] = 0.9
        perturbed_batch = self.mask_perturbation.perturbe(unsupervised_batch, logits)
        self._print_sentence(perturbed_batch)

    def test_swap_perturbation(self):
        self.swap_perturbation = load_perturbation_from_descriptor("swap_noname_1", self.device, self.tokenizer)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        input_ids, input_mask, loss_mask, segment_ids = unsupervised_batch
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 9)
        # All tokens are no names
        logits[:, :, 0] = 1
        perturbed_batch = self.swap_perturbation.perturbe(unsupervised_batch, logits)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)

    def test_char_remove_perturbation(self):
        self.char_remove_perturbation = load_perturbation_from_descriptor("char_remove_0.5", self.device, self.tokenizer)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        input_ids, input_mask, loss_mask, segment_ids = unsupervised_batch
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 9)
        # All tokens are names
        logits[:, :, 1] = 1
        perturbed_batch = self.char_remove_perturbation.perturbe(unsupervised_batch, logits)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)

    def test_mask_word_replace_perturbation(self):
        self.mask_word_replace_perturbation = load_perturbation_from_descriptor("mask_word_replace_0.5", self.device, self.tokenizer)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        input_ids, input_mask, loss_mask, segment_ids = unsupervised_batch
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 9)
        # All tokens are no names
        logits[:, :, 0] = 1
        perturbed_batch = self.mask_word_replace_perturbation.perturbe(unsupervised_batch, logits)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)

    def test_word_replace_perturbation(self):
        self.word_replace_perturbation = load_perturbation_from_descriptor("word_replace_0.5", self.device, self.tokenizer)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        input_ids, input_mask, loss_mask, segment_ids = unsupervised_batch
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 9)
        # All tokens are no names
        logits[:, :, 0] = 1
        perturbed_batch = self.word_replace_perturbation.perturbe(unsupervised_batch, logits)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)

    def test_filled_perturbation(self):
        self.filled_perturbation = load_perturbation_from_descriptor("filled_0.5", self.device, self.tokenizer)
        unsupervised_batch = next(iter(self.unsupervised_dataloader))
        input_ids, input_mask, loss_mask, segment_ids = unsupervised_batch
        logits = torch.zeros(input_ids.shape[0], input_ids.shape[1], 9)
        perturbed_batch = self.filled_perturbation.perturbe(unsupervised_batch, logits)
        self._print_sentence(unsupervised_batch)
        self._print_sentence(perturbed_batch)
