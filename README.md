# DiSF
(ICLR Oral) DiSF Combatting Dimensional Collapse in LLM Pre-training Data via Diversified File Selection.


# Step 0 Environment and Data Installation
### Environment
We provide the detailed environment in environment.txt file.  

### Data Install(SlimPajama-627B)
You can download SlimPajama-627B through following command:
>cd /path/to/dataset  
git lfs install  
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B  

or try other sources  
>git clone https://gitee.com/hf-datasets/SlimPajama-627B

### Data pre-processing
You should first tokenize the datasets and divide them into chunks:  
>python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split validation --percentage 1.0  
python scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split train --percentage 1.0

In parallel, you need to extract text features:


### Data Selection for DISF
Step 1) 
