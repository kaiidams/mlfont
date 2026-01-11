from . import bdffont
import numpy as np
from jsonargparse import auto_cli


class Main:
    DEFAULT_TARGET = "-Shinonome-Gothic-Medium-R-Normal--16-150-75-75-C-80-ISO10646-1"

    def predict(
        self,
        *,
        batch_size: int = 1,
        data_type: str = "morning_light_bdf",
        ckpt_path: str,
        output: str,
        timestep_skip: int = 1,
        predict_target: str = DEFAULT_TARGET,
    ) -> None:
        r"""Generate font with FontMaskModel."""
        from tqdm import tqdm
        import torch
        from .fontmask import FontMaskDataModule, FontMaskModel

        data = FontMaskDataModule(
            kernel=(2,2),
            batch_size=batch_size,
            download=True,
            data_type=data_type,
            predict_target=predict_target,
        )
        data.setup("predict")

        model = FontMaskModel.load_from_checkpoint(
            ckpt_path,
            kernel=(2,2),
            timestep_skip=timestep_skip,
        )
        model.eval()
        model_predict_step = torch.compile(
            torch.no_grad(model.predict_step)
        )

        with open(output, "w") as fp:
            for batch_idx, batch in tqdm(enumerate(data.predict_dataloader())):
                outputs = model_predict_step(batch, batch_idx)
                charsets = outputs["charset"]
                encodings = outputs["encoding"]
                image = outputs["image"]
                for sample_idx in range(len(image)):
                    charset = charsets[sample_idx]
                    encoding = encodings[sample_idx]
                    image_hex = "".join(["%02X" % x for x in image[sample_idx].tobytes()])
                    bbh, bbw = image[sample_idx].shape
                    fp.write(f"{charset}\t{encoding}\t{bbw}\t{bbh}\t{image_hex}\n")

    def display(
        self,
        *,
        input: str,
    ) -> None:
        r"""Display the generated font glyphs."""
        with open(input) as fp:
            for line in fp:
                parts = line.rstrip().split("\t")
                charset, encoding, bbw, bbh, image_hex = parts
                encoding = int(encoding)
                bbw = int(bbw)
                bbh = int(bbh)
                image = np.array(
                    [int(image_hex[i : i + 2], 16) for i in range(0, len(image_hex), 2)],
                    dtype=np.uint8,
                ).reshape(bbh, bbw)
                print(f"Charset: {charset}")
                print(f"Encoding: U+{encoding:04x} Char: {chr(encoding)}")
                for row in image:
                    line = "".join(["[]" if x > 0 else ". " for x in row])
                    print(line)

    def makebdf(
        self,
        *,
        input: str,
        output: str,
    ) -> None:
        r"""Export as a BDF file."""
        font = bdffont.load("data/local/shnm8x16u.bdf")
        font.font = font.font.replace("-Shinonome-Gothic-", "-ML-Fixed-")
        font.properties["foundry"] = "ML"
        font.properties["family_name"] = "Fixed"
        with open(input) as fp:
            for line in fp:
                parts = line.rstrip().split("\t")
                charset, encoding, bbw, bbh, image_hex = parts
                encoding = int(encoding)
                bbw = int(bbw)
                bbh = int(bbh)
                image = np.array(
                    [int(image_hex[i : i + 2], 16) for i in range(0, len(image_hex), 2)],
                    dtype=np.uint8,
                ).reshape(bbh, bbw)
                image = np.minimum(image > 0, 1)
                bitmap_width = image.shape[1]
                bitmap_height = image.shape[0]
                bitmap_rowbytes = (bitmap_width + 7) // 8
                mask = (1 << (7 - np.arange(8)))
                x = np.sum(image.reshape(-1, 8) * mask[None, :], axis=-1).astype(np.uint8)
                bitmap = x.tobytes()
                assert len(bitmap) == 16, len(bitmap)
                # TODO support different geometry.
                char = bdffont.BDFChar(
                    f"U+{encoding:04x}",
                    encoding=encoding,
                    bbx=(8, 16, 0, -2),
                    swidth=(480, 0),
                    dwidth=(8, 0),
                    bitmap=bitmap,
                )
                font.chars.append(char)
        bdffont.save(font, output)

    def makeshnmu(
        self,
        *,
        input: str = "data/ShinonomeBDFFontDataset/raw/shinonome-0.9.11/bdf/",
        output: str = "data/local",
    ) -> None:
        r"""Make Unicode Shinonome fonts."""

        from . import bdffont
        import os

        files = [
            "shnm6x12a.bdf",
            "shnm7x14a.bdf",
            "shnm8x16a.bdf",
            "shnm9x18a.bdf",
        ]
        names = [
            "-Shinonome-Gothic-Medium-R-Normal--12-110-75-75-C-60-ISO10646-1",
            "-Shinonome-Gothic-Medium-R-Normal--14-130-75-75-C-70-ISO10646-1",
            "-Shinonome-Gothic-Medium-R-Normal--16-150-75-75-C-80-ISO10646-1",
            "-Shinonome-Gothic-Medium-R-Normal--18-170-75-75-C-90-ISO10646-1",
        ]

        M = {}
        for cp in range(256):
            if 0x80 <= cp < 0xa1 or 0xe0 <= cp < 0x100:
                continue
            if 0x00 <= cp < 0x80 and cp != 0x7e:
                continue
            if cp == 0x7e:
                ch = chr(0x203e)
            else:
                ch = bytes([cp]).decode("Windows-31J")
            M[cp] = ch
            # print("%02x" % cp, "U+%04x" % ord(ch), ch)

        for file, name in zip(files, names):
            ufont = bdffont.load(os.path.join(input, file))
            rfont = bdffont.load(os.path.join(input, file.replace("a.bdf", "r.bdf")))
            ufont.font = name
            ufont.properties["charset_registry"] = "ISO10646"
            x = []
            for ch in rfont.chars:
                cp = ch.encoding
                if cp in M:
                    ch.name = "%04x" % ord(M[cp])
                    ch.encoding = ord(M[cp])
                    x.append(ch)
                    ch.bitmap = rfont.tobytes(ch)
                    ufont.chars.append(ch)
            bdffont.save(ufont, os.path.join(output, file.replace("a.bdf", "u.bdf")))

if __name__ == "__main__":
    auto_cli(Main)
