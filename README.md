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

Download generated BDF font [fonts-ml-0.1.tar.gz](./fonts-ml-0.1.tar.gz).
