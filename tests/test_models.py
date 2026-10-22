# Copyright (c) Katsuya Iida.  All Rights Reserved.
# See LICENSE in the project root for license information.

import unittest

import numpy as np

try:
    import torch
except ImportError:
    raise unittest.SkipTest("torch is not available")

from mlfont.models import (
    FontMaskDataModule,
    FontMaskModel,
    FontMaskTokenizer,
)


class TestFontMask(unittest.TestCase):

    def test_tokenizer(self):
        tokenizer = FontMaskTokenizer(kernel=(2, 2))
        images = [
            np.zeros((7, 5), dtype=np.uint8),
            np.zeros((13, 8), dtype=np.uint8),
        ]
        type_ids = [1, 2]
        inputs = tokenizer.encode_batch(images, type_ids)
        self.assertEqual(len(inputs), 5)
        self.assertEqual(inputs["token"].shape, (2, 256))
        self.assertEqual(inputs["xpos"].shape, (2, 256))
        self.assertEqual(inputs["ypos"].shape, (2, 256))
        self.assertEqual(inputs["type"].shape, (2, 256))
        self.assertEqual(inputs["mask"].shape, (2, 256))
        self.assertTrue(np.all(inputs["mask"][0] == (np.arange(256) >= 4 * 3 + 1)))
        self.assertTrue(np.all(inputs["mask"][1] == (np.arange(256) >= 7 * 4 + 1)))

    def test_tokenizer_list(self):
        tokenizer = FontMaskTokenizer(kernel=(2, 2))
        images = [
            [
                np.zeros((7, 5), dtype=np.uint8),
                np.zeros((13, 8), dtype=np.uint8),
            ],
            [
                np.zeros((10, 6), dtype=np.uint8),
                np.zeros((13, 8), dtype=np.uint8),
                np.zeros((18, 10), dtype=np.uint8),
            ],
        ]
        type_ids = [[1, 2], [3, 4, 5]]
        inputs = tokenizer.encode_list_batch(images, type_ids, max_length=90)
        self.assertEqual(len(inputs), 5)
        self.assertEqual(inputs["token"].shape, (2, 90))
        self.assertEqual(inputs["xpos"].shape, (2, 90))
        self.assertEqual(inputs["ypos"].shape, (2, 90))
        self.assertEqual(inputs["type"].shape, (2, 90))
        self.assertEqual(inputs["mask"].shape, (2, 90))
        self.assertTrue(np.all(inputs["mask"][0] == (np.arange(90) >= 1 + (4 * 3 + 1) + (7 * 4 + 1))))
        self.assertTrue(np.all(inputs["mask"][1] == np.zeros(90, dtype=np.bool_)))

    def test_data(self):
        FontMaskDataModule

    def test_model(self):
        FontMaskModel


if __name__ == "__main__":
    unittest.main()
