#### pre-train command
```bash
lightning run model   --accelerator=cuda     --devices 8  \pretrain/tinyllama.py \
--devices 8 --train_data_dir /your_tokenized_train_path \
--val_data_dir /your_tokenized_val_path
```

notably, there are parameter to change in \pretrain/tinyllama.py  
We provide three set of parameters in the inital several lines of tinyllama.py  
Besides, the prefix name in train_data_config should be the same with your tokenized files in ./scripts/prepare_slimpajama.py
