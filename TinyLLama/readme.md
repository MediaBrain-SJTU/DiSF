#### pre-train command
```bash
lightning run model   --accelerator=cuda     --devices 8  \pretrain/tinyllama.py \
--devices 8 --train_data_dir /your_tokenized_train_path \
--val_data_dir /your_tokenized_val_path
```
