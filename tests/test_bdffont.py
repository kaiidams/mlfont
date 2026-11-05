# Copyright (c) Katsuya Iida.  All Rights Reserved.
# See LICENSE in the project root for license information.

import os
import tempfile
import unittest

from mlfont import bdffont

SAMPLE_PATH = os.path.join(
    os.path.dirname(__file__), "nil2.bdf"
)


class TestBDFFontLoadSave(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_load(self):
        font = bdffont.load(SAMPLE_PATH)
        self.assertIsInstance(font, bdffont.BDFFont)
        self.assertEqual(font.version, "2.1")
        self.assertEqual(
            font.font, "-misc-nil-medium-r-normal--2-20-75-75-c-10-misc-fontspecific"
        )
        self.assertEqual(font.size, (2, 75, 75))
        self.assertEqual(font.fontboundingbox, (1, 2, 0, 0))
        self.assertEqual(len(font.chars), 224)

    def test_save(self):
        font = bdffont.load(SAMPLE_PATH)
        out_path = os.path.join(self.temp_dir.name, "saved.bdf")
        bdffont.save(font, out_path)
        self.assertTrue(os.path.exists(out_path))

        reloaded = bdffont.load(out_path)
        self.assertEqual(reloaded.version, font.version)
        self.assertEqual(reloaded.font, font.font)
        self.assertEqual(reloaded.size, font.size)
        self.assertEqual(reloaded.fontboundingbox, font.fontboundingbox)
        self.assertEqual(len(reloaded.chars), len(font.chars))
        for orig_ch, rel_ch in zip(font.chars, reloaded.chars):
            self.assertEqual(orig_ch.name, rel_ch.name)
            self.assertEqual(orig_ch.encoding, rel_ch.encoding)
            self.assertEqual(orig_ch.bbx, rel_ch.bbx)
            self.assertEqual(orig_ch.dwidth, rel_ch.dwidth)
            self.assertEqual(font.tobytes(orig_ch), reloaded.tobytes(rel_ch))


if __name__ == "__main__":
    unittest.main()
