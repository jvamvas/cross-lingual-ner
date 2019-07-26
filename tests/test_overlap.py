from unittest import TestCase

from pytorch_pretrained_bert import BertTokenizer

from scripts.diagnose_ner import CoNLL2003Dataset
from scripts.overlap import OverlapMeasure


class OverlapTestCase(TestCase):

    def setUp(self) -> None:
        self.source = CoNLL2003Dataset("../tests/data/source.txt")
        self.target = CoNLL2003Dataset("../tests/data/target.txt")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased", do_lower_case=False)
        self.overlap_measure = OverlapMeasure(self.source, self.target, self.tokenizer)

    def test_get_coverage(self):
        print(self.overlap_measure.get_word_type_coverage())
        print(self.overlap_measure.get_word_token_coverage())
        print(self.overlap_measure.get_name_type_coverage())
        print(self.overlap_measure.get_name_token_coverage())
