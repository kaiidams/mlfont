# Copyright (c) Katsuya Iida.  All Rights Reserved.
# See LICENSE in the project root for license information.

import unittest

import numpy as np

from mlfont.datasets import (
    make_random_image,
    random_resize_image,
)


class TestDatasetUtils(unittest.TestCase):

    def test_make_random_image(self):
        min_w, max_w = 4, 12
        min_h, max_h = 6, 24
        image = make_random_image(min_width=min_w, max_width=max_w,
                                  min_height=min_h, max_height=max_h)
        self.assertTrue(np.issubdtype(image.dtype, np.integer))
        self.assertTrue((image == 0).any() or (image == 255).any())
        self.assertTrue(min_w <= image.shape[1] <= max_w)
        self.assertTrue(min_h <= image.shape[0] <= max_h)

    def test_random_resize_image(self):
        original = np.arange(12 * 8).reshape((12, 8)).astype(np.uint8)

        # Make a image smaller
        target_w, target_h = 4, 6
        resized = random_resize_image(original, (target_w, target_h))
        self.assertEqual(resized.shape, (target_h, target_w))

        # Make a image smaller with non integer scale factor
        target_w, target_h = 5, 11
        resized = random_resize_image(original, (target_w, target_h))
        self.assertEqual(resized.shape, (target_h, target_w))

        # Make a image larger with non integer scale factor
        target_w, target_h = 14, 25
        resized = random_resize_image(original, (target_w, target_h))
        self.assertEqual(resized.shape, (target_h, target_w))


if __name__ == "__main__":
    unittest.main()
