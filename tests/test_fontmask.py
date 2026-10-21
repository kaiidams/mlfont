# Copyright (c) Katsuya Iida.  All Rights Reserved.
# See LICENSE in the project root for license information.

import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch is not available")

from mlfont.fontmask import (
    FontMaskDataModule,
    FontMaskModel,
    FontMaskTokenizer,
)


class TestFontMask(unittest.TestCase):

    def test_tokenizer(self):
        FontMaskTokenizer

    def test_data(self):
        FontMaskDataModule

    def test_model(self):
        FontMaskModel


if __name__ == "__main__":
    unittest.main()
