import os
from unittest import TestCase

from scripts.convert_iob_to_iob2 import convert_iob_to_iob2


class IOB2TestCase(TestCase):

    def setUp(self) -> None:
        self.iob1_file = os.path.join("data", "sample.iob1")
        self.iob2_file = os.path.join("data", "sample.iob2")
        self.iob2_ref_file = os.path.join("data", "sample.iob2.ref")

    def test_convert_iob_to_iob2(self):
        convert_iob_to_iob2(self.iob1_file, self.iob2_file)
        with open(self.iob2_file) as f, open(self.iob2_ref_file) as ref:
            self.assertEqual(f.read(), ref.read())
