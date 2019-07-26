from unittest import TestCase

from pytorch_pretrained_bert import BertTokenizer

from scripts.maximize_coverage import maximize_coverage
from scripts.conll_subset import CoNLL2003Dataset
from scripts.sample_fixed_numtokens import get_subset


class OverlapTestCase(TestCase):

    def setUp(self) -> None:
        self.source = CoNLL2003Dataset("../data/conll03_multilingual/valid.en")
        self.target = CoNLL2003Dataset("../data/conll03_multilingual/valid.de")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)

    def test_get_coverage(self):
        output = maximize_coverage(self.source, self.target, 100, self.tokenizer)
        print(output)

    def test_sample_fixed_numtokens(self):
        output = get_subset(self.source, 100, 3570)
        print(output)
