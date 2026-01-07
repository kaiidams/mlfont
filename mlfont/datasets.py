# Copyright (c) Katsuya Iida.  All Rights Reserved.
# See LICENSE in the project root for license information.

import logging
import os
import random
import sys
from glob import glob
from typing import Any, NamedTuple
from collections.abc import Callable, Iterator

import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset

from . import bdffont

logger = logging.getLogger(__name__)


DEFAULT_X11_SUBSETS = {
    "font-jis-misc-1.0.3.tar.bz2",
    "font-misc-misc-1.1.2.tar.bz2",
    "font-sony-misc-1.0.3.tar.bz2",
    "font-arabic-misc-1.0.3.tar.bz2",
    "font-daewoo-misc-1.0.3.tar.bz2",
    "font-bh-100dpi-1.0.3.tar.bz2",
    # MUTT fonts covers Unicode special characters.
    # "font-mutt-misc-1.0.3.tar.bz2",
    "font-adobe-75dpi-1.0.3.tar.bz2",
    # This is for cursors.
    # "font-sun-misc-1.0.3.tar.bz2",
    "font-isas-misc-1.0.3.tar.bz2",
    "font-adobe-100dpi-1.0.3.tar.bz2",
    "font-winitzki-cyrillic-1.0.3.tar.bz2",
    # This is too small.
    # "font-micro-misc-1.0.3.tar.bz2",
    "font-screen-cyrillic-1.0.4.tar.bz2",
    "font-bitstream-75dpi-1.0.3.tar.bz2",
    "font-misc-cyrillic-1.0.3.tar.bz2",
    "font-schumacher-misc-1.1.2.tar.bz2",
    "font-bitstream-100dpi-1.0.3.tar.bz2",
    "font-bh-lucidatypewriter-100dpi-1.0.3.tar.bz2",
    # This is for cursors.
    # "font-dec-misc-1.0.3.tar.bz2",
    "font-bh-75dpi-1.0.3.tar.bz2",
    "font-adobe-utopia-75dpi-1.0.4.tar.bz2",
    "font-cronyx-cyrillic-1.0.3.tar.bz2",
    # This is for cursors.
    # "font-cursor-misc-1.0.3.tar.bz2",
    "font-adobe-utopia-100dpi-1.0.4.tar.bz2",
    "font-bh-lucidatypewriter-75dpi-1.0.3.tar.bz2",
}

DEFAULT_SHINONOME_SUBSETS = {
    "shnm6x12a.bdf",
    "shnm6x12r.bdf",
    "shnm7x14a.bdf",
    "shnm7x14r.bdf",
    "shnm8x16a.bdf",
    "shnm8x16r.bdf",
    "shnmk12.bdf",
    # "shnmk12min.bdf",
    "shnmk14.bdf",
    # "shnmk14min.bdf",
    "shnmk16.bdf",
    # "shnmk16min.bdf",
}


class X11BDFFontDataset(IterableDataset):
    def __init__(
        self,
        root: str,
        download: bool = False,
        subsets: set[str] | None = DEFAULT_X11_SUBSETS,
        no_cjk_unicode: bool = True,
        no_cjk_wide: bool = True,
        font_filter: Callable[[bdffont.BDFFont], bool] | None = None,
    ) -> None:
        self.root = root
        self.no_cjk_unicode = no_cjk_unicode
        self.no_cjk_wide = no_cjk_wide
        self.font_filter = font_filter
        self.fonts = {}
        if download:
            self._download_checksums()
        self._load_resources(subsets)
        logger.info(f"Found {len(self.resources)} resources.")
        if download:
            for file, md5 in self.resources:
                self._download_archive(file, md5)
        self.files = self._collect_bdffiles()
        if subsets is None:
            logger.info(f"Found {len(self.files)} BDF files for all subsets.")
        else:
            logger.info(
                f"Found {len(self.files)} BDF files for {len(subsets)} subsets."
            )

    mirrors = [
        "https://www.x.org/releases/X11R7.7/src/font/",
    ]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _download_checksums(self):
        from torchvision.datasets.utils import download_url

        download_url(
            self.mirrors[0] + "/CHECKSUMS",
            self.raw_folder,
            "CHECKSUMS",
            "ed881d2082156618a0d53ffaf3cc4a5f",
        )

    def _load_resources(self, subsets: set[str] | None) -> None:
        resources: list[tuple[str, str]] = []
        with open(os.path.join(self.raw_folder, "CHECKSUMS")) as fp:
            for line in fp:
                line = line.strip()
                if line.startswith("MD5:") and line.endswith(".bz2"):
                    _, md5, file = line.split()
                    if subsets is None or file in subsets:
                        resources.append((file, md5))
        self.resources = resources

    def _download_archive(self, file: str, md5: str):
        from torchvision.datasets.utils import download_and_extract_archive

        download_and_extract_archive(
            self.mirrors[0] + file,
            download_root=self.raw_folder,
            md5=md5,
        )

    def _collect_bdffiles(self) -> list[str]:
        files = []
        for dirname, _ in self.resources:
            if dirname.endswith(".bz2"):
                dirname, _ = os.path.splitext(dirname)
            if dirname.endswith(".tar"):
                dirname, _ = os.path.splitext(dirname)
            for file in glob(
                os.path.join(self.raw_folder, dirname, "**/*.bdf"),
                recursive=True,
            ):
                files.append(file)
        return files

    @staticmethod
    def _check_cjk_wide(font) -> bool:
        charset_registry = font.properties.get("charset_registry")
        charset_registry, _, _ = charset_registry.partition(".")
        return charset_registry in {"KSC5601", "GB2312", "JISX0208", "JISX0201"}

    @staticmethod
    def _check_cjk_unicode(font) -> bool:
        return font.properties.get("add_style_name") in {"cn", "zh", "ja", "ko"}

    def __iter__(self) -> Iterator[tuple[bdffont.BDFFont, bdffont.BDFChar]]:
        for file in self.files:
            try:
                font = bdffont.load(file)
                if self.no_cjk_wide and self._check_cjk_wide(font):
                    logger.debug("Skipping CJK font: %s", file)
                    continue
                if self.no_cjk_unicode and self._check_cjk_unicode(font):
                    logger.debug("Skipping CJK Unicode font: %s", file)
                    continue
                if self.font_filter and not self.font_filter(font):
                    logger.debug("Skipping filtered font: %s", file)
                    continue
                for char in font.chars:
                    yield font, char
            except Exception as e:
                raise RuntimeError(f"Failed to load {file}.") from e


class ShinonomeBDFFontDataset(IterableDataset):
    def __init__(
        self,
        root: str,
        download: bool = False,
        subsets: set[str] | None = DEFAULT_SHINONOME_SUBSETS,
    ) -> None:
        self.root = root
        self.fonts = {}
        self.resources = [
            ("shinonome-0.9.11p1.tar.bz2", "5fb94de9a9971ac67a4d53d62f77bc1d"),
        ]

        logger.info(f"Found {len(self.resources)} resources.")
        if download:
            for file, md5 in self.resources:
                self._download_archive(file, md5)
        self.files = self._collect_bdffiles(subsets)
        if subsets is None:
            logger.info(f"Found {len(self.files)} BDF files for all subsets.")
        else:
            logger.info(
                f"Found {len(self.files)} BDF files for {len(subsets)} subsets."
            )

    mirrors = [
        "http://openlab.ring.gr.jp/efont/dist/shinonome/",
    ]

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, self.__class__.__name__, "raw")

    def _download_archive(self, file: str, md5: str):
        from torchvision.datasets.utils import download_and_extract_archive

        download_and_extract_archive(
            self.mirrors[0] + file,
            download_root=self.raw_folder,
            md5=md5,
        )

    def _collect_bdffiles(self, subsets: set[str] | None) -> list[str]:
        files = []
        for dirname, _ in self.resources:
            if dirname.endswith(".bz2"):
                dirname, _ = os.path.splitext(dirname)
            if dirname.endswith(".tar"):
                dirname, _ = os.path.splitext(dirname)
            if dirname.endswith("p1"):
                dirname = dirname[:-2]
            for file in glob(
                os.path.join(self.raw_folder, dirname, "**/*.bdf"),
                recursive=True,
            ):
                name = os.path.basename(file)
                if subsets is None or name in subsets:
                    files.append(file)
        return files

    def __iter__(self) -> Iterator[tuple[bdffont.BDFFont, bdffont.BDFChar]]:
        for file in self.files:
            try:
                font = bdffont.load(file)
                for char in font.chars:
                    yield font, char
            except Exception as e:
                raise RuntimeError(f"Failed to load {file}.") from e


class LocalBDFFontDataset(IterableDataset):
    def __init__(
        self,
        root: str,
        subsets: set[str] | None = None,
    ) -> None:
        self.root = root
        self.fonts = {}
        self.files = glob("**/*.bdf", root_dir=root, recursive=True)
        if subsets:
            self.files = [
                file for file in self.files if os.path.basename(file) in subsets
            ]
        if not self.files:
            raise ValueError("No files found.")

        logger.info(f"Found {len(self.files)} files.")

    def __iter__(self) -> Iterator[tuple[bdffont.BDFFont, bdffont.BDFChar]]:
        for file in self.files:
            try:
                font = bdffont.load(os.path.join(self.root, file))
                for char in font.chars:
                    yield font, char
            except Exception as e:
                raise RuntimeError(f"Failed to load {file}.") from e


GroupKey = Any


class PropIdImage(NamedTuple):
    prop_id: str
    encoding: int
    image: np.ndarray


def _charset_encoding(
    font: bdffont.BDFFont, char: bdffont.BDFChar
) -> tuple[str, int] | None:
    charreg = str(font.properties["charset_registry"])
    charenc = str(font.properties["charset_encoding"])
    charreg, _, _ = charreg.partition(".")  # Remove year, e.g. JISX0208.1983
    charset = sys.intern(f"{charreg}-{charenc}")
    if charset == "ISO8859-1":
        # ISO8859-1 is compatible with ISO10646-1.
        charset = "ISO10646-1"
    encoding = char.encoding

    # 'ISO10646-1': 402,
    # 'KOI8-R': 82,
    # 'ISO8859-1': 54,
    # 'ISO646.1991-IRV': 30,
    # 'Adobe-FontSpecific': 12,
    # 'FontSpecific-0': 5,
    # 'DEC-DECtech': 4,
    # 'KSC5601.1987-0': 3,
    # 'GB2312.1980-0': 3,
    # 'JISX0208.1983-0': 3,
    # 'JISX0201.1976-0': 2,

    if isinstance(encoding, tuple):
        return None
    if charset in {"ISO10646-1", "ISO8859-1"}:
        if encoding <= 0x20 or 0x7F <= encoding < 0xA0:
            return None
    elif charset == "Misc-FontSpecific":
        return None
    return (charset, encoding)


def _prop_id_image(font: bdffont.BDFFont, char: bdffont.BDFChar) -> PropIdImage:
    assert isinstance(char.encoding, int)
    return PropIdImage(font.font, char.encoding, font.numpy(char))


def _remove_duplicates(items: list[PropIdImage]) -> None:
    visited = [items[-1].image]
    for i in range(len(items) - 2, -1, -1):
        x = items[i].image
        for y in visited:
            if x.shape == y.shape and np.all(x == y):
                items.pop(i)
                break


def groupby_fonts(
    iter,
    key_fn: Callable[[bdffont.BDFFont, bdffont.BDFChar], GroupKey],
    value_fn: Callable[[bdffont.BDFFont, bdffont.BDFChar], PropIdImage],
    remove_duplicates: bool,
    min_items: int,
) -> list[tuple[GroupKey, list[PropIdImage]]]:
    data: dict[GroupKey, list[PropIdImage]] = {}
    font, char = None, None
    try:
        for font, char in iter:
            k = key_fn(font, char)
            if k is None:
                continue
            if k not in data:
                data[k] = []
            data[k].append(value_fn(font, char))
    except Exception as e:
        raise RuntimeError(f"Failed to load dataset. {font}/{char}") from e

    result = [(k, v) for k, v in data.items()]
    if remove_duplicates:
        for _, items in result:
            _remove_duplicates(items)
    return [(k, v) for k, v in result if len(v) >= min_items]


class FontDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        iter,
        transform: Callable[[PropIdImage], Any],
        value_fn: Callable[
            [bdffont.BDFFont, bdffont.BDFChar], PropIdImage
        ] = _prop_id_image,
    ) -> None:
        self.transform = transform
        self.data = [value_fn(*x) for x in iter]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.transform(self.data[idx])


class GroupedFontDataset(Dataset):
    def __init__(
        self,
        iter,
        transform: Callable[[GroupKey, list[PropIdImage], PropIdImage | None], Any],
        key_fn: Callable[
            [bdffont.BDFFont, bdffont.BDFChar], GroupKey
        ] = _charset_encoding,
        value_fn: Callable[
            [bdffont.BDFFont, bdffont.BDFChar], PropIdImage
        ] = _prop_id_image,
        remove_duplicates: bool = True,
        min_items: int = 1,
        predict_target: str | None = None,
    ) -> None:
        self.repeat_count = 1
        self.grouped_data = groupby_fonts(
            iter, key_fn, value_fn, remove_duplicates, min_items
        )
        if predict_target:
            self.data_index = [
                (key_idx, -1)
                for key_idx, (_, values) in enumerate(self.grouped_data)
                if not any(predict_target == value.prop_id for value in values)
            ]
        else:
            self.data_index = [
                (key_idx, value_idx)
                for key_idx, (_, prop_id_image) in enumerate(self.grouped_data)
                for value_idx, _ in enumerate(prop_id_image)
            ]
        self.transform = transform

    def __len__(self):
        return len(self.data_index) * self.repeat_count

    def __getitem__(self, idx: int):
        key_idx, value_idx = self.data_index[idx % len(self.data_index)]
        key, values = self.grouped_data[key_idx]
        src_values = list(values)
        if value_idx == -1:
            tgt_value = None
        else:
            tgt_value = src_values.pop(value_idx)
        return self.transform(key, src_values, tgt_value)

    def repeat(
        self,
        repeat: int | None,
        size: int | None,
    ):
        if repeat is None:
            if size is None:
                self.repeat_count = 1
            elif size < len(self):
                self.data_index = random.sample(self.data_index, k=size)
            else:
                self.repeat_count = max(1, size // len(self))
        else:
            if size is not None:
                raise ValueError("Either size or repeat can be given.")
            self.repeat_count = repeat
        return self

    def shuffle(self):
        random.shuffle(self.data_index)
        return self


def random_resize_image(input: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if input.size == 0:
        return input
    oldh, oldw = input.shape[-2:]
    neww, newh = size
    if neww <= 1 or newh <= 1:
        return input

    s = random.randrange(256)
    e = oldh * 256 - 1 - random.randrange(256)
    x = input
    if newh >= oldh:
        i = ((newh - 1) * s + (e - s) * np.arange(newh)) // (256 * (newh - 1))
        x = x[..., i, :]
    else:
        i = (newh * s + (e - s) * np.arange(newh + 1)) // (256 * newh)
        x = x[..., i[1:] - 1, :] + x[..., i[:-1], :]

    s = random.randrange(256)
    e = oldw * 256 - 1 - random.randrange(256)
    if neww >= oldw:
        i = ((neww - 1) * s + (e - s) * np.arange(neww)) // (256 * (neww - 1))
        x = x[..., i]
    else:
        i = (neww * s + (e - s) * np.arange(neww + 1)) // (256 * neww)
        x = x[..., i[1:] - 1] + x[..., i[:-1]]
    return x


def make_random_image(
    min_bbw: int = 4,
    max_bbw: int = 12,
    min_bbh: int = 6,
    max_bbh: int = 24,
    dot_prob: float = 0.2,
    block_prob: float = 0.8,
    line_prob: float = 0.8,
):
    bbw = random.randint(min_bbw, max_bbw)
    bbh = random.randint(min_bbh, max_bbh)
    image = np.random.random(size=[bbh, bbw]) < dot_prob
    image = image.astype(np.uint8)
    if random.random() < block_prob:
        x0 = random.randint(0, bbw - 1)
        y0 = random.randint(0, bbh - 1)
        x1 = random.randint(0, bbw - 1)
        y1 = random.randint(0, bbh - 1)
        c = 1 if random.random() < dot_prob else 0
        if x0 > x1:
            x0, x1 = x1, x0
        if y0 > y1:
            y0, y1 = y1, y0
        image[y0 : y1 + 1, x0 : x1 + 1] = c
    if random.random() < line_prob:
        x0 = random.randint(0, bbw - 1)
        y0 = random.randint(0, bbh - 1)
        if random.random() < 0.5:
            x1 = random.randint(0, bbw - 1)
            if x0 > x1:
                x0, x1 = x1, x0
            image[y0, x0 : x1 + 1] = 1
        else:
            y1 = random.randint(0, bbh - 1)
            if y0 > y1:
                y0, y1 = y1, y0
            image[y0 : y1 + 1, x0] = 1
    return image * 255
