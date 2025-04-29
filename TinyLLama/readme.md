#### pre-train command
```bash
lightning run model   --accelerator=cuda     --devices 8  \pretrain/tinyllama.py \
--devices 8 --train_data_dir /your_tokenized_train_path \
--val_data_dir /your_tokenized_val_path
```

notably, there are parameter to change in \pretrain/tinyllama.py
1) We provide three set of parameters in the inital several lines of tinyllama.py. For example:  

```bash
#1.1B:
model_name = "tiny_LLaMA_1b"
name = "tinyllama_1b_disf_16_1024"
out_dir = Path("out") / name
num_of_devices = 8
global_batch_size=512
learning_rate = 4e-4
micro_batch_size = 4
max_step=100000
warmup_steps=500
log_step_interval = 10
eval_iters = 100
save_step_interval=1000
eval_step_interval=1000
#checkpoint_path = "yourpath/out/xx/iter-240000-ckpt.pth"
```
2) Besides, the prefix name in train_data_config should be the same with your tokenized files in ./scripts/prepare_slimpajama.py  
