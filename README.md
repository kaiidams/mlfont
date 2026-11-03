# Morning Light Font

Morning Light Font is an experimental generated font with machine learning.

All fonts included in the datasets are public domain X11 fonts,
misc fonts and Shinonome fonts. 

`FontMaskModel` is a Transformer-based masked vision model which predicts
masked tokens of the target font conditioned by the source fonts. The model
generates an 8x16 font with all characters included in the dataset.

Please note that this project is experimental. The generated font doesn't
have acceptable quality. Here is the comparison.

The generated 8x13 glyphs.

![ml8x13](./ml8x13.png)

The original X11 8x13 glyphs.

![8x13](./8x13.png)

Some of missing glyphs in 8x13 are available in ml8x13.

## BDF font

Download generated BDF font [fonts-ml-0.1.tar.gz](https://github.com/kaiidams/mlfont/releases/download/v0.1/fonts-ml-0.1.tar.gz).

## Metrics

Name | X Height | Cap Height | Descent | Ascent | Base
--- | --- | --- | --- | --- | ---
`6x12.bdf` | 5 | 7 | 2 | 10
`6x13.bdf` | 6 | 9 | 2 | 11
`7x14.bdf` | 7 | 10 | 2 | 12
`9x18.bdf` | 7 | 10 | 4 | 14
`10x20.bdf` | 8 | 13 | 4 | 16
`ml6x12.bdf` | 6 | 9 | 2 | 10 | `6x13.bdf`
`ml7x14.bdf` | 7 | 10 | 2 | 12 | `7x14.bdf`
`ml8x16.bdf` | 8 | 11 | 3 | 13 | `8x13.bdf`
`ml9x18.bdf` | 8 | 12 | 3 | 15 | `9x18.bdf`
`ml10x20.bdf` | 9 | 13 | 4 | 16 | `10x20.bdf`