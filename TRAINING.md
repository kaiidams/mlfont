# Training

## Generate resized BDF fonts

Vertically resamples and adjusts metrics of X11 misc BDF fonts
to generate 11 Morning Light (ml*) BDF files.

```sh
python -m mlfont.cli make_resized
```

## Edit manually BDF fonts

Edit generated BDF files. In my experiment, I edited only
characters included in ISO-8859-1 of `ml6x12.bdf`, `ml8x15.bdf`, `ml9x18.bdf`
and `ml10x20.bdf`, and removed all characters which are not included in
ISO-8859-1. `ml7x14.bdf` was untouched.

## Training

I used regular-weight X11 font-misc-misc BDF fonts and edited BDF fonts.

```sh
data_type=morning_light_bdf
python -m mlfont.fontmask fit 
    --data.data_type=$data_type \
    --data.train_size=300000 \
    --data.batch_size=128 \
    --data.download=True \
    --model.hidden_dim=512 \
    --model.lr=1e-4 \
    --trainer.max_epochs=500 \
    --trainer.precision=16
```

## Prediction

This generates text files containing predicted bitmaps for
characters not included in the training data. The file is used to
recreate a BDF file with edited characters.

```sh
ckpt_path=./tmp/fontmask-20261112.ckpt
python -m mlfont.cli predict \
    --ckpt_path=$ckpt_path \
    --output=./tmp/20261112/output-ml6x12.txt \
    --batch_size=4 \
    --predict_target=ml6x12 \
```

### Display the result

This displays predicted bitmaps in ASCII text.

```sh
python -m mlfont.cli display \
    --input=./tmp/20261112/output-ml6x12.txt \
```

Output:

```
Charset: ISO10646-1
Encoding: U+0100 Char: Ā
[][][][][]. 
. . . . . . 
. . []. . . 
. []. []. . 
[]. . . []. 
[]. . . []. 
[][][][][]. 
[]. . . []. 
[]. . . []. 
[]. . . []. 
. . . . . . 
. . . . . . 
Charset: ISO10646-1
Encoding: U+0101 Char: ā
. . . . . . 
...
```

### Export BDF files

This generates a BDF file with both the original bitmaps
and predicted bitmaps.

```sh
python -m mlfont.cli export \
    --input=./tmp/20261112/output-ml6x12.txt \
    --base=./data/local/ml6x12.bdf \
    --output ./tmp/20261112/ml6x12.bdf
```
