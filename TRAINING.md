# Training

## Generate resized BDF fonts

These fonts fill the gap between font sizes.
Generates `8x16.bdf`, `9x16.bdf` and `10x18.bdf`.

```sh
python -m mlfont.cli makeresized
```

## Training

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

```sh
ckpt_path=./tmp/fontmask-20260131.ckpt
python -m mlfont.cli predict \
    --ckpt_path=$ckpt_path \
    --output=./tmp/output-7x14.txt \
    --batch_size=4 \
    --predict_target=7x14 \
```
