# Copyright (c) Katsuya Iida.  All Rights Reserved.
# See LICENSE in the project root for license information.

import logging
import random
from typing import cast

from mlfont.datasets import (
    GroupedFontDataset,
    PropIdImage,
    ShinonomeBDFFontDataset,
    X11BDFFontDataset,
    LocalBDFFontDataset,
    random_resize_image,
    make_random_image,
)
import lightning as L
from torch import nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from torchmetrics.classification import Accuracy

logger = logging.getLogger(__name__)

SHINONOME_BDF_V2_INPUT_SPECS = [
    ("-Shinonome-Gothic-Medium-R-Normal--12-110-75-75-C-60-ISO8859-1", 6, 12),
    ("-Shinonome-Gothic-Medium-R-Normal--12-110-75-75-C-60-JISX0201.1976-0", 6, 12),
    ("-Shinonome-Gothic-Medium-R-Normal--14-130-75-75-C-70-ISO8859-1", 7, 14),
    ("-Shinonome-Gothic-Medium-R-Normal--14-130-75-75-C-70-JISX0201.1976-0", 7, 14),
    ("-Shinonome-Gothic-Medium-R-Normal--16-150-75-75-C-80-ISO8859-1", 8, 16),
    ("-Shinonome-Gothic-Medium-R-Normal--16-150-75-75-C-80-JISX0201.1976-0", 8, 16),
    ("-Shinonome-Gothic-Medium-R-Normal--18-170-75-75-C-90-ISO8859-1", 9, 18),
    ("-Shinonome-Gothic-Medium-R-Normal--18-170-75-75-C-90-JISX0201.1976-0", 9, 18),
    ("-Shinonome-Gothic-Medium-R-Normal--12-110-75-75-C-120-JISX0208.1990-0", 12, 12),
    ("-Shinonome-Gothic-Medium-R-Normal--14-130-75-75-C-140-JISX0208.1990-0", 14, 14),
    ("-Shinonome-Gothic-Medium-R-Normal--16-150-75-75-C-160-JISX0208.1990-0", 16, 16),
]

X11_BDF_V2_INPUT_SPECS = [
    ("-Misc-Fixed-Medium-R-Normal--6-60-75-75-C-40-ISO10646-1", 4, 6),
    ("-Misc-Fixed-Medium-R-Normal--7-70-75-75-C-50-ISO10646-1", 5, 7),
    ("-Misc-Fixed-Medium-R-Normal--8-80-75-75-C-50-ISO10646-1", 5, 8),
    ("-Misc-Fixed-Medium-R-Normal--9-90-75-75-C-60-ISO10646-1", 6, 9),
    ("-Misc-Fixed-Medium-R-Normal--10-100-75-75-C-60-ISO10646-1", 6, 10),
    ("-Misc-Fixed-Medium-R-SemiCondensed--12-110-75-75-C-60-ISO10646-1", 6, 12),
    ("-Misc-Fixed-Medium-R-SemiCondensed--13-120-75-75-C-60-ISO10646-1", 6, 13),
    ("-Misc-Fixed-Medium-R-Normal--13-120-75-75-C-70-ISO10646-1", 7, 13),
    ("-Misc-Fixed-Medium-R-Normal--14-130-75-75-C-70-ISO10646-1", 7, 14),
    ("-Misc-Fixed-Medium-R-Normal--13-120-75-75-C-80-ISO10646-1", 8, 13),
    ("-Misc-Fixed-Medium-R-Normal--15-140-75-75-C-90-ISO10646-1", 9, 15),
    ("-Misc-Fixed-Medium-R-Normal--18-120-100-100-C-90-ISO10646-1", 9, 18),
    ("-Misc-Fixed-Medium-R-Normal--20-200-75-75-C-100-ISO10646-1", 10, 20),
    ("-Shinonome-Gothic-Medium-R-Normal--16-150-75-75-C-80-ISO8859-1", 8, 16),
]

MORNING_LIGHT_BDF_V2_INPUT_SPECS = [
    ("-Misc-Fixed-Medium-R-Normal--6-60-75-75-C-40-ISO10646-1", 4, 6),
    ("-Misc-Fixed-Medium-R-Normal--7-70-75-75-C-50-ISO10646-1", 5, 7),
    ("-Misc-Fixed-Medium-R-Normal--8-80-75-75-C-50-ISO10646-1", 5, 8),
    ("-Misc-Fixed-Medium-R-Normal--9-90-75-75-C-60-ISO10646-1", 6, 9),
    ("-Misc-Fixed-Medium-R-Normal--10-100-75-75-C-60-ISO10646-1", 6, 10),
    ("-Misc-Fixed-Medium-R-SemiCondensed--12-110-75-75-C-60-ISO10646-1", 6, 12),
    ("-Misc-Fixed-Medium-R-SemiCondensed--13-120-75-75-C-60-ISO10646-1", 6, 13),
    ("-Misc-Fixed-Medium-R-Normal--13-120-75-75-C-70-ISO10646-1", 7, 13),
    ("-Misc-Fixed-Medium-R-Normal--14-130-75-75-C-70-ISO10646-1", 7, 14),
    ("-Misc-Fixed-Medium-R-Normal--13-120-75-75-C-80-ISO10646-1", 8, 13),
    ("-Misc-Fixed-Medium-R-Normal--15-140-75-75-C-90-ISO10646-1", 9, 15),
    ("-Misc-Fixed-Medium-R-Normal--18-120-100-100-C-90-ISO10646-1", 9, 18),
    ("-Misc-Fixed-Medium-R-Normal--20-200-75-75-C-100-ISO10646-1", 10, 20),
    ("-Shinonome-Gothic-Medium-R-Normal--16-150-75-75-C-80-ISO10646-1", 8, 16),
]


class FontMaskTokenizer:
    """Tokenizer for BDF font bitmaps that generates a token for each block."""

    def __init__(
        self,
        kernel: tuple[int, int] = (3, 3),
    ) -> None:
        self.kernel = kernel
        k = 1 << (kernel[0] * kernel[1])
        self.vocab_size = k + 4
        self.cls_id = k
        self.sep_id = k + 1
        self.mask_id = k + 2
        self.pad_id = k + 3

    def encode_list_batch(
        self,
        images: list[list[np.ndarray]],
        type_ids: list[list[int]],
        max_length: int = 256,
    ):
        data = {"token": [], "xpos": [], "ypos": [], "type": []}
        for image, type_id in zip(images, type_ids):
            for k, v in self.encode_list(image, type_id, max_length=max_length).items():
                data[k].append(v)
        data = {
            k: self._padarray(
                v, self.pad_id if k == "token" else 0, max_length=max_length
            )
            for k, v in data.items()
        }
        data["mask"] = data["token"] == self.pad_id
        return data

    def encode_batch(
        self, images: list[np.ndarray], type_ids: list[int], max_length: int = 256
    ):
        data = {"token": [], "xpos": [], "ypos": [], "type": []}
        zero_a = np.array([0], dtype=np.int64)
        cls_a = np.array([self.cls_id], dtype=np.int64)
        for image, type_id in zip(images, type_ids):
            for k, v in self.encode(image, type_id).items():
                v = np.insert(v, 0, cls_a if k == "token" else zero_a)
                data[k].append(v)
        data = {
            k: self._padarray(
                v, self.pad_id if k == "token" else 0, max_length=max_length
            )
            for k, v in data.items()
        }
        data["mask"] = data["token"] == self.pad_id
        return data

    def _padarray(self, x: list[np.ndarray], pad_value: int = 0, max_length: int = 256):
        z = np.full([len(x), max_length], pad_value, dtype=x[0].dtype)
        for i, y in enumerate(x):
            z[i, : y.shape[0]] = y
        return z

    def _splitarray(self, x: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
        return [y[ymask] for y, ymask in zip(x, mask)]

    def decode_batch(
        self,
        token: np.ndarray,
        xpos: np.ndarray,
        ypos: np.ndarray,
        type_id: np.ndarray,
        mask: np.ndarray,
    ) -> list[np.ndarray]:
        data = {
            "token": self._splitarray(token, ~mask),
            "xpos": self._splitarray(xpos, ~mask),
            "ypos": self._splitarray(ypos, ~mask),
            "type": self._splitarray(type_id, ~mask),
        }
        images = []
        for token_, xpos_, ypos_ in zip(
            data["token"],
            data["xpos"],
            data["ypos"],
        ):
            token_ = token_[1:]
            xpos_ = xpos_[1:]
            ypos_ = ypos_[1:]
            image = self.decode(token=token_, xpos=xpos_, ypos=ypos_)
            images.append(image)
        return images

    def decode_list_batch(
        self,
        token: np.ndarray,
        xpos: np.ndarray,
        ypos: np.ndarray,
        type_id: np.ndarray,
        mask: np.ndarray,
    ):
        data = {
            "token": self._splitarray(token, ~mask),
            "xpos": self._splitarray(xpos, ~mask),
            "ypos": self._splitarray(ypos, ~mask),
            "type": self._splitarray(type_id, ~mask),
        }
        del type_id
        images = []
        type_ids = []
        for token_, xpos_, ypos_, type_id in zip(
            data["token"],
            data["xpos"],
            data["ypos"],
            data["type"],
        ):
            image, type_id_ = self.decode_list(
                token=token_, xpos=xpos_, ypos=ypos_, type_id=type_id
            )
            images.append(image)
            type_ids.append(type_id_)
        return images, type_ids

    def decode_list(
        self,
        token: np.ndarray,
        xpos: np.ndarray,
        ypos: np.ndarray,
        type_id: np.ndarray,
    ) -> tuple[list[np.ndarray], list[int]]:
        (sep_pos,) = np.where(token == self.sep_id)
        start = 1
        images = []
        type_ids = []
        for end in sep_pos:
            image = self.decode(
                token=token[start:end], xpos=xpos[start:end], ypos=ypos[start:end]
            )
            images.append(image)
            type_ids.append(type_id[start] - 1)
            start = end + 1
        return images, type_ids

    def encode_list(
        self,
        images: list[np.ndarray],
        type_ids: np.ndarray | list[int],
        max_length: int = 256,
    ) -> dict[str, np.ndarray]:
        zero_a = np.array([0], dtype=np.int64)
        cls_a = np.array([self.cls_id], dtype=np.int64)
        sep_a = np.array([self.sep_id], dtype=np.int64)
        data = {
            "token": [cls_a],
            "xpos": [zero_a],
            "ypos": [zero_a],
            "type": [zero_a],
        }
        datalen = 1
        for image, type_id in zip(images, type_ids):
            x = self.encode(image, type_id)
            for k, v in data.items():
                v.append(np.asarray(x[k]))
            data["token"].append(sep_a)
            data["xpos"].append(zero_a)
            data["ypos"].append(zero_a)
            type_a = np.full([1], type_id, dtype=np.int64)
            data["type"].append(type_a)
            datalen += cast(np.ndarray, x["token"]).ndim + 1
            if datalen >= max_length:
                break
        return {k: np.concatenate(v) for k, v in data.items()}

    def encode(
        self, image: np.ndarray, type_id: np.ndarray | int
    ) -> dict[str, np.ndarray]:
        """Tokenize a 2D numpy array of a bitmap image."""
        assert image.ndim == 2
        kernel = self.kernel
        bbh, bbw = image.shape
        out_h = (bbh + kernel[0] - 1) // kernel[0]
        out_w = (bbw + kernel[1] - 1) // kernel[1]
        pad_h = out_h * kernel[0] - bbh
        pad_w = out_w * kernel[1] - bbw
        x = np.pad(image, ((0, pad_h), (0, pad_w)))
        x = (x > 0).astype(np.uint32)
        x = x.reshape(out_h, kernel[0], out_w, kernel[1])
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(out_h * out_w, kernel[0] * kernel[1])
        x = np.sum(x << (np.arange(kernel[0] * kernel[1])[None, :]), axis=1)
        token: np.ndarray = x.astype(np.int64)

        x = np.arange(out_h, dtype=np.int64)
        x = np.repeat(x[:, None], out_w, 1) + 1
        ypos = x.reshape(-1)

        x = np.arange(out_w, dtype=np.int64)
        x = np.repeat(x[None, :], out_h, 0) + 1
        xpos = x.reshape(-1)

        type_id = np.full_like(token, type_id + 1)

        return {
            "token": token,
            "xpos": xpos,
            "ypos": ypos,
            "type": type_id,
        }

    def decode(
        self, token: np.ndarray, xpos: np.ndarray, ypos: np.ndarray
    ) -> np.ndarray:
        """Untokenize a 2D numpy array of a bitmap image."""
        kernel = self.kernel
        assert isinstance(token, np.ndarray)
        x: np.ndarray = (
            token[:, None].astype(np.uint32)
            >> np.arange(kernel[0] * kernel[1])[None, :]
        ) & 1
        sel = token >= self.cls_id
        x[sel, :] = token[sel, None] + 2 - self.cls_id
        x = x.astype(np.uint8)
        out_w: int = np.max(xpos)
        out_h: int = np.max(ypos)
        x = x.reshape(out_h, out_w, kernel[0], kernel[1])
        x = x.transpose(0, 2, 1, 3)
        x = x.reshape(out_h * kernel[0], out_w * kernel[1])
        return x


def _get_input_spec(data_type: str):
    if data_type == "shinonome_bdf":
        return SHINONOME_BDF_V2_INPUT_SPECS
    elif data_type == "x11_bdf":
        return X11_BDF_V2_INPUT_SPECS
    elif data_type == "morning_light_bdf":
        return MORNING_LIGHT_BDF_V2_INPUT_SPECS
    else:
        raise ValueError(f"Unknown data_type: {data_type}")


class FontMaskDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root: str = "./data",
        download: bool = False,
        train_size: int | None = None,
        train_repeat: int | None = None,
        val_size: int | None = None,
        val_repeat: int | None = None,
        batch_size: int = 256,
        data_type: str = "x11_bdf",
        predict_target: str | None = None,
        src_resize_prob: float = 0.2,
        src_drop_prob: float = 0.2,
        tgt_random_prob: float = 0.25,
        split_train_test: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self._fit_ready = False
        self._predict_ready = False
        self.data_root = data_root
        self.download = download
        self.train_size = train_size
        self.train_repeat = train_repeat
        self.val_size = val_size
        self.val_repeat = val_repeat
        self.batch_size = batch_size
        self.train_repeat = train_repeat
        self.data_type = data_type
        self.predict_target = predict_target
        self.src_resize_prob = src_resize_prob
        self.src_drop_prob = src_drop_prob
        self.tgt_random_prob = tgt_random_prob
        self.split_train_test = split_train_test
        self.input_specs = _get_input_spec(data_type)
        self.max_width = max(bbw for _, bbw, _ in self.input_specs)
        self.max_height = max(bbh for _, _, bbh in self.input_specs)
        self.tokenizer = FontMaskTokenizer()
        self.max_src_length, self.max_tgt_length = self._compute_max_sequence_length()
        self.prop_map = {
            prop_id: prop_idx
            for prop_idx, (prop_id, _, _) in enumerate(self.input_specs)
        }

    def _compute_max_sequence_length(self):
        src_len = 1
        tgt_len = 1
        kernel = self.tokenizer.kernel
        for _, bbw, bbh in self.input_specs:
            out_w = (bbw + kernel[1] - 1) // kernel[1]
            out_h = (bbh + kernel[0] - 1) // kernel[0]
            src_len += out_w * out_h + 1
            tgt_len = max(tgt_len, out_w * out_h + 1)
        return src_len, tgt_len

    def prepare_data(self) -> None:
        pass  # self._create_dataset()

    def _create_dataset(self):
        if self.data_type == "shinonome_bdf":
            dataset = ShinonomeBDFFontDataset(
                self.data_root,
                download=self.download,
            )
        elif self.data_type == "x11_bdf":
            dataset = X11BDFFontDataset(
                self.data_root,
                download=self.download,
                no_cjk_wide=True,
                no_cjk_unicode=True,
                subsets={
                    "font-misc-misc-1.1.2.tar.bz2",
                },
            )
        elif self.data_type == "morning_light_bdf":
            all_fonts = {prop_id for prop_id, _, _ in self.input_specs}
            dataset = torch.utils.data.ChainDataset(
                [
                    LocalBDFFontDataset(
                        self.data_root,
                        subsets={
                            "shnm8x16u.bdf",
                        },
                    ),
                    X11BDFFontDataset(
                        self.data_root,
                        download=self.download,
                        no_cjk_wide=True,
                        no_cjk_unicode=True,
                        font_filter=lambda font: font.font in all_fonts,
                        subsets={
                            "font-misc-misc-1.1.2.tar.bz2",
                        },
                    ),
                ]
            )
        else:
            raise ValueError(f"Unknown data_type: '{self.data_type}'.")

        return dataset

    def setup(self, stage: str) -> None:
        if stage == "fit":
            self._setup_fit()
        elif stage == "predict":
            self._setup_predict()
        else:
            raise ValueError(f"Unknown stage: {stage}.")

    def _transform_fn(
        self,
        _key,
        src_values: list[PropIdImage],
        tgt_value: PropIdImage | None,
    ) -> dict[str, np.ndarray | list[np.ndarray] | int]:
        assert tgt_value is not None
        src_images: list[np.ndarray | None] = [None] * len(self.input_specs)
        if random.random() < self.tgt_random_prob:
            tgt_idx = random.randrange(0, len(self.input_specs))
            _, bbw, bbh = self.input_specs[tgt_idx]
            tgt_image = make_random_image(bbw, bbw, bbh, bbh)
            tgt_synth = 1
        else:
            tgt_idx = self.prop_map[tgt_value.prop_id]
            tgt_image = tgt_value.image
            tgt_synth = 0
            for src_value in src_values:
                src_idx = self.prop_map[src_value.prop_id]
                src_image = src_value.image
                src_images[src_idx] = src_image

        for src_idx in range(len(src_images)):
            if src_idx == tgt_idx:
                src_images[src_idx] = None
            elif random.random() < self.src_resize_prob:
                _, bbw, bbh = self.input_specs[src_idx]
                src_images[src_idx] = random_resize_image(tgt_image, (bbw, bbh))
            elif random.random() < self.src_drop_prob:
                src_images[src_idx] = None

        tgt_image = (tgt_image > 0).astype(np.uint8)
        src_type_ids = np.array(
            [
                src_idx
                for src_idx, src_image in enumerate(src_images)
                if src_image is not None
            ],
            dtype=np.int64,
        )
        np.random.shuffle(src_type_ids)
        src_images = [src_images[src_idx] for src_idx in src_type_ids]

        return {
            "src_images": cast(list[np.ndarray], src_images),
            "src_types": src_type_ids,
            "tgt_image": tgt_image,
            "tgt_type": tgt_idx,
            "tgt_synth": tgt_synth,
        }

    def _predict_transform_fn(
        self,
        _key,
        src_values: list[PropIdImage],
        tgt_value: PropIdImage | None,
    ) -> dict[str, np.ndarray | list[np.ndarray] | int]:
        assert tgt_value is None
        assert self.predict_target is not None
        src_images: list[np.ndarray | None] = [None] * len(self.input_specs)
        tgt_idx = self.prop_map[self.predict_target]
        _, bbw, bbh = self.input_specs[tgt_idx]
        tgt_image = np.zeros([bbh, bbw], dtype=np.uint8)
        tgt_synth = 0
        for src_value in src_values:
            src_idx = self.prop_map[src_value.prop_id]
            src_image = src_value.image
            src_images[src_idx] = src_image

        src_type_ids = np.array(
            [
                src_idx
                for src_idx, src_image in enumerate(src_images)
                if src_image is not None
            ],
            dtype=np.int64,
        )
        np.random.shuffle(src_type_ids)
        src_images = [src_images[src_idx] for src_idx in src_type_ids]

        return {
            "charset": _key[0],
            "encoding": _key[1],
            "src_images": cast(list[np.ndarray], src_images),
            "src_types": src_type_ids,
            "tgt_image": tgt_image,
            "tgt_type": tgt_idx,
            "tgt_synth": tgt_synth,
        }

    def _setup_fit(self) -> None:
        if self._fit_ready:
            logger.info("DataModule is already ready.")
            return

        dataset = self._create_dataset()
        if self.split_train_test:
            from sklearn.model_selection import train_test_split

            train_data, val_data = train_test_split(
                list(dataset), test_size=0.2, random_state=42, shuffle=True
            )
        else:
            train_data = dataset
            val_data = dataset

        self.train_data = GroupedFontDataset(
            train_data,
            transform=self._transform_fn,
            remove_duplicates=False,
            min_items=1,
        ).repeat(repeat=self.train_repeat, size=self.train_size)

        self.val_data = (
            GroupedFontDataset(
                val_data,
                transform=self._transform_fn,
                remove_duplicates=False,
                min_items=1,
            )
            .shuffle()
            .repeat(repeat=self.val_repeat, size=self.val_size)
        )

        self._fit_ready = True

    def _setup_predict(self):
        prop_id = None
        for prop_id, _, _ in self.input_specs:
            if prop_id == self.predict_target:
                break
        if self._predict_ready:
            logger.info("DataModule is already ready.")
            return

        dataset = self._create_dataset()

        self.predict_data = GroupedFontDataset(
            dataset,
            transform=self._predict_transform_fn,
            remove_duplicates=False,
            min_items=1,
            predict_target=prop_id,
        )

        self._predict_ready = True

    def _collate_fn(self, batch):
        src_image = [x["src_images"] for x in batch]
        tgt_image = [x["tgt_image"] for x in batch]
        src_type = [x["src_types"] for x in batch]
        tgt_type = [x["tgt_type"] for x in batch]
        tgt_synth = torch.tensor([x["tgt_synth"] for x in batch], dtype=torch.long)
        src_data = self.tokenizer.encode_list_batch(
            src_image,
            src_type,
            max_length=self.max_src_length,
        )
        tgt_data = self.tokenizer.encode_batch(
            tgt_image,
            tgt_type,
            max_length=self.max_tgt_length,
        )
        src_data = {"src_" + k: torch.from_numpy(v) for k, v in src_data.items()}
        tgt_data = {"tgt_" + k: torch.from_numpy(v) for k, v in tgt_data.items()}
        tgt_data["tgt_synth"] = tgt_synth
        # these are not tensors, they are cast torch.Tensor.
        if "name" in batch[0]:
            src_data["name"] = cast(torch.Tensor, [x["name"] for x in batch])
        if "charset" in batch[0]:
            src_data["charset"] = cast(torch.Tensor, [x["charset"] for x in batch])
        if "encoding" in batch[0]:
            src_data["encoding"] = cast(torch.Tensor, [x["encoding"] for x in batch])

        return src_data | tgt_data

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=3,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=3,
        )

    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_data,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=3,
        )


class FontMaskModel(L.LightningModule):
    r"""Masked Discrete Diffusion Models."""

    def __init__(
        self,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_sequence_length: int = 256,
        gamma: float = 0.99,
        src_mask_prob: float = 0.1,
        num_timesteps: int = 200,
        timestep_skip: int = 1,
        data_type: str = "x11_bdf",
        lr: float = 1e-4,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.hidden_dim = hidden_dim
        self.gamma = gamma
        self.src_mask_prob = src_mask_prob
        self.num_timesteps = num_timesteps
        self.timestep_skip = timestep_skip
        self.input_specs = _get_input_spec(data_type=data_type)
        self.max_width = max(bbw for _, bbw, _ in self.input_specs)
        self.max_height = max(bbh for _, _, bbh in self.input_specs)
        self.tokenizer = FontMaskTokenizer()
        self.max_sequence_length = max_sequence_length

        self.xpos_embedding = nn.Embedding(
            self.max_width + 1,
            self.hidden_dim,
            padding_idx=0,
        )
        self.ypos_embedding = nn.Embedding(
            self.max_height + 1,
            self.hidden_dim,
            padding_idx=0,
        )
        self.token_embedding = nn.Embedding(
            self.tokenizer.vocab_size,
            self.hidden_dim,
            padding_idx=self.tokenizer.pad_id,
        )
        self.type_embedding = nn.Embedding(
            len(self.input_specs) + 1,
            self.hidden_dim,
            padding_idx=0,
        )
        self.tgt_synth_embedding = nn.Embedding(
            2,  # tgt_synth 0, 1
            self.hidden_dim,
        )
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=num_heads,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=hidden_dim * 4,
            norm_first=False,
            batch_first=True,
        )
        self.proj_out = nn.Linear(self.hidden_dim, self.tokenizer.vocab_size)
        self.criterion = nn.CrossEntropyLoss(reduction="none")
        self.train_acc = Accuracy(
            task="multiclass", num_classes=self.tokenizer.vocab_size
        )
        self.train_pixel_acc = Accuracy(
            task="binary",
        )
        self.valid_acc = Accuracy(
            task="multiclass", num_classes=self.tokenizer.vocab_size
        )
        self.valid_pixel_acc = Accuracy(
            task="binary",
        )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)  # pyright: ignore[reportAttributeAccessIssue]
        return optimizer

    def forward(
        self,
        src_token: torch.Tensor,
        src_xpos: torch.Tensor,
        src_ypos: torch.Tensor,
        src_type: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_token: torch.Tensor,
        tgt_xpos: torch.Tensor,
        tgt_ypos: torch.Tensor,
        tgt_type: torch.Tensor,
        tgt_mask: torch.Tensor,
        tgt_synth: torch.Tensor,
    ) -> torch.Tensor:
        assert src_token.ndim == 2
        assert src_mask.ndim == 2
        assert tgt_token.ndim == 2
        assert tgt_mask.ndim == 2

        src = self.token_embedding(src_token)
        src = src + self.xpos_embedding(src_xpos)
        src = src + self.ypos_embedding(src_ypos)
        src = src + self.type_embedding(src_type)
        tgt = self.token_embedding(tgt_token)
        tgt = tgt + self.xpos_embedding(tgt_xpos)
        tgt = tgt + self.ypos_embedding(tgt_ypos)
        tgt = tgt + self.type_embedding(tgt_type)
        tgt = tgt + self.tgt_synth_embedding(torch.unsqueeze(tgt_synth, dim=1))
        x = self.transformer(
            src,
            tgt,
            src_key_padding_mask=src_mask,
            tgt_key_padding_mask=tgt_mask,
            memory_key_padding_mask=src_mask,
            src_is_causal=False,
            tgt_is_causal=False,
        )
        # x: [B, N, D]
        x = self.proj_out(x)
        return x

    @torch.no_grad
    def mask_tokens(self, token: torch.Tensor, mask_prob: float = 0.0) -> torch.Tensor:
        timestep = torch.randint(
            1, self.num_timesteps + 1, [token.size(0)], device=token.device
        )
        mask = torch.rand(token.shape, device=token.device) >= torch.unsqueeze(
            self.gamma**timestep, -1
        )
        if mask_prob > 0:
            mask = mask & (
                torch.rand([token.size(0), 1], device=token.device) < mask_prob
            )
        mask = mask & (token < self.tokenizer.cls_id)
        return torch.masked_fill(token, mask, self.tokenizer.mask_id)

    def rec_loss(
        self,
        logits: torch.Tensor,
        label: torch.Tensor,
        mask: torch.Tensor,
        phase: str,
    ) -> torch.Tensor:
        loss = self.criterion(torch.transpose(logits, 1, 2), label)
        loss = torch.masked_fill(loss, mask, 0.0)
        loss = torch.sum(loss)
        not_mask = torch.logical_not(mask)
        samples = torch.sum(not_mask)
        loss = loss / samples
        self.log(f"{phase}_loss", loss, prog_bar=True)
        return loss

    def compute_accuracy(
        self, logits: torch.Tensor, label: torch.Tensor, mask: torch.Tensor, phase: str
    ):
        with torch.no_grad():
            preds = torch.argmax(logits, dim=-1).to(dtype=label.dtype)
            not_mask = torch.logical_not(mask)
            kernel = self.tokenizer.kernel
            pixel_preds = (
                torch.unsqueeze(preds, -1)
                & (1 << torch.arange(kernel[0] * kernel[1], device=preds.device))
            ).to(torch.bool)
            pixel_label = (
                torch.unsqueeze(label, -1)
                & (1 << torch.arange(kernel[0] * kernel[1], device=preds.device))
            ).to(torch.bool)
        if phase == "train":
            self.train_acc(preds[not_mask], label[not_mask])
            self.train_pixel_acc(pixel_preds[not_mask], pixel_label[not_mask])
            self.log("train_acc", self.train_acc, on_step=True, on_epoch=False)
            self.log(
                "train_pixel_acc", self.train_pixel_acc, on_step=True, on_epoch=False
            )
        elif phase == "val":
            self.valid_acc(preds[not_mask], label[not_mask])
            self.valid_pixel_acc(pixel_preds[not_mask], pixel_label[not_mask])
            self.log("valid_acc", self.valid_acc, on_step=False, on_epoch=True)
            self.log(
                "valid_pixel_acc", self.valid_pixel_acc, on_step=False, on_epoch=True
            )

    def compute_loss(
        self, batch: dict[str, torch.Tensor], phase: str
    ) -> dict[str, torch.Tensor]:
        torch.autograd.set_detect_anomaly(True)
        inputs = dict(batch)
        src_token = inputs["src_token"]
        lbl_token = inputs["tgt_token"]
        lbl_mask = inputs["tgt_mask"]
        src_token = self.mask_tokens(src_token, mask_prob=self.src_mask_prob)
        tgt_token = self.mask_tokens(lbl_token, mask_prob=0.0)
        inputs["src_token"] = src_token
        inputs["tgt_token"] = tgt_token
        logits = self.forward(**inputs)
        # logits: [B, N, C]
        loss = self.rec_loss(logits, lbl_token, lbl_mask, phase)
        self.compute_accuracy(logits, lbl_token, lbl_mask, phase)
        return {
            "loss": loss,
            "logits": logits,
            "src_token": src_token,
            "tgt_token": tgt_token,
        }

    def training_step(self, batch, batch_idx):
        outputs = self.compute_loss(batch, "train")
        return outputs["loss"]

    def validation_step(self, batch, batch_idx):
        return self.compute_loss(batch, "val")

    def on_train_epoch_start(self):
        scheduler = self.lr_schedulers()
        if scheduler:
            assert not isinstance(scheduler, list)
            lr = scheduler.get_last_lr()[0]
            self.log("lr", lr)

    def on_validation_batch_end(self, outputs, batch, batch_idx, dataloader_idx=0):
        if batch_idx > 0:
            return

        for logger in self.loggers:
            with torch.no_grad():
                grid = self.make_grid(batch, cast(dict[str, torch.Tensor], outputs))

            if hasattr(logger, "experiment"):
                logger.experiment.add_image(  # type: ignore[attr-defined]
                    "reconstructed_images", grid, self.current_epoch
                )

    def denoise_setup(self, inputs: dict[str, torch.Tensor]) -> None:
        # Mask all
        tgt_token = torch.full_like(inputs["tgt_token"], self.tokenizer.pad_id)
        tgt_token.masked_fill_(~inputs["tgt_mask"], self.tokenizer.mask_id)
        tgt_token[:, 0] = self.tokenizer.cls_id
        inputs["tgt_token"] = tgt_token
        inputs["timestep"] = torch.full(
            [tgt_token.size(0)],
            self.num_timesteps,
            dtype=torch.long,
            device=tgt_token.device,
        )

    @torch.no_grad()
    def denoise_step(
        self,
        inputs: dict[str, torch.Tensor],
        timestep_skip: int = 1,
        eagar: bool = True,
    ) -> dict[str, torch.Tensor]:
        timestep = inputs.pop("timestep")
        logits = self(**inputs)
        # probability of special tokens are 0.
        logits[:, :, -4:] = float("-inf")
        tgt_token = inputs["tgt_token"]

        if eagar:
            pred_logits, pred_token = torch.max(logits, -1)
            pred_logits.masked_fill_(tgt_token != self.tokenizer.mask_id, float("-inf"))
            maxlogits_pos = torch.argmax(pred_logits, -1, keepdim=True)
            unmask = (
                torch.arange(pred_logits.shape[-1], device=pred_logits.device)
                == maxlogits_pos
            )
        else:
            gamma_cumprod = torch.unsqueeze(self.gamma**timestep, -1)
            unmask_prob = (
                (1 / (self.gamma**timestep_skip) - 1)
                * gamma_cumprod
                / (1 - gamma_cumprod)
            )
            pred_token = torch.argmax(logits, -1)
            r = torch.rand_like(logits[:, :, 0])
            unmask = (r < unmask_prob) & (tgt_token == self.tokenizer.mask_id)
        tgt_token = torch.where(unmask, pred_token, tgt_token)
        mask_count = torch.sum(tgt_token == self.tokenizer.mask_id)
        outputs = {
            "logits": logits,
            "src_token": inputs["src_token"],
            "tgt_token": inputs["tgt_token"],
            "mask_count": mask_count,
        }
        inputs["tgt_token"] = tgt_token
        inputs["timestep"] = timestep - timestep_skip
        return outputs

    def predict_step(self, batch: dict[str, torch.Tensor], batch_idx: int):
        charset = batch.pop("charset")
        encoding = batch.pop("encoding")
        inputs = dict(batch)
        self.denoise_setup(inputs)
        outputs = None
        assert self.num_timesteps // self.timestep_skip > 0
        for _ in range(self.num_timesteps // self.timestep_skip):
            outputs = self.denoise_step(inputs, timestep_skip=self.timestep_skip)
            if outputs["mask_count"].item() == 0:
                break
        assert outputs
        pred_image = self.tokenizer.decode_batch(
            token=inputs["tgt_token"].cpu().numpy(),
            xpos=batch["tgt_xpos"].cpu().numpy(),
            ypos=batch["tgt_ypos"].cpu().numpy(),
            type_id=batch["tgt_type"].cpu().numpy(),
            mask=batch["tgt_mask"].cpu().numpy(),
        )
        types = batch["tgt_type"][:, 1].cpu().numpy() - 1
        image = []
        for type_, image_ in zip(types, pred_image):
            _, bbw, bbh = self.input_specs[type_]
            assert bbw <= image_.shape[1]
            assert bbh <= image_.shape[0]
            image.append(image_[:bbh, :bbw])

        return {
            "type": batch["tgt_type"],
            "charset": charset,
            "encoding": encoding,
            "image": image,
        }

    def make_grid(
        self, batch: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        import torchvision

        batch = batch.copy()

        src_images, src_types = self.tokenizer.decode_list_batch(
            token=outputs["src_token"].cpu().numpy(),
            xpos=batch["src_xpos"].cpu().numpy(),
            ypos=batch["src_ypos"].cpu().numpy(),
            type_id=batch["src_type"].cpu().numpy(),
            mask=batch["src_mask"].cpu().numpy(),
        )
        tgt_images = self.tokenizer.decode_batch(
            token=batch["tgt_token"].cpu().numpy(),
            xpos=batch["tgt_xpos"].cpu().numpy(),
            ypos=batch["tgt_ypos"].cpu().numpy(),
            type_id=batch["tgt_type"].cpu().numpy(),
            mask=batch["tgt_mask"].cpu().numpy(),
        )
        masked_images = self.tokenizer.decode_batch(
            token=outputs["tgt_token"].cpu().numpy(),
            xpos=batch["tgt_xpos"].cpu().numpy(),
            ypos=batch["tgt_ypos"].cpu().numpy(),
            type_id=batch["tgt_type"].cpu().numpy(),
            mask=batch["tgt_mask"].cpu().numpy(),
        )
        pred_token = torch.argmax(outputs["logits"], -1)
        pred_images = self.tokenizer.decode_batch(
            token=pred_token.cpu().numpy(),
            xpos=batch["tgt_xpos"].cpu().numpy(),
            ypos=batch["tgt_ypos"].cpu().numpy(),
            type_id=batch["tgt_type"].cpu().numpy(),
            mask=batch["tgt_mask"].cpu().numpy(),
        )

        num_items = min(32, len(src_images))
        grid_image = np.full(
            [num_items, 4, 3, self.max_height, self.max_width], 127, dtype=np.uint8
        )

        palette = np.array(
            [
                [0, 0, 0],
                [255, 255, 255],
                [255, 0, 0],  # [CLS]
                [255, 0, 255],  # [SEP]
                [0, 255, 0],  # [MASK]
                [0, 0, 255],  # [PAD]
            ]
        )

        def draw_image(image: np.ndarray, idx, typ):
            bbh = min(image.shape[0], self.max_height)
            bbw = min(image.shape[1], self.max_width)
            grid_image[idx, typ, :, :bbh, :bbw] = np.transpose(
                palette[image], [2, 0, 1]
            )[:, :bbh, :bbw]

        for idx in range(num_items):
            if src_images[idx]:
                src_image = src_images[idx][0]
                src_type = src_types[idx][0]
                _, bbw, bbh = self.input_specs[src_type]
                kernel = self.tokenizer.kernel
                w = (bbw + kernel[1] - 1) // kernel[1] * kernel[1]
                h = (bbh + kernel[0] - 1) // kernel[0] * kernel[0]
                assert src_image.shape == (h, w), [src_image.shape, h, w, src_type]
                src_image = src_image[:bbh, :bbw]
                draw_image(src_image, idx, 0)
            draw_image(tgt_images[idx], idx, 1)
            draw_image(masked_images[idx], idx, 2)
            draw_image(pred_images[idx], idx, 3)

        x = torch.from_numpy(grid_image)
        x = torch.flatten(x, 0, 1)
        grid = torchvision.utils.make_grid(x, nrow=12)
        return grid


def cli_main():
    from lightning.pytorch.cli import LightningCLI

    logging.basicConfig(level=logging.INFO)

    class FontMaskLightningCLI(LightningCLI):
        def add_arguments_to_parser(self, parser):
            parser.link_arguments("data.data_type", "model.data_type")

    cli = FontMaskLightningCLI(  # noqa: F841
        FontMaskModel,
        FontMaskDataModule,
    )


if __name__ == "__main__":
    cli_main()
