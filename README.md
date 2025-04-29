# DiSF
(ICLR Oral) DiSF Combatting Dimensional Collapse in LLM Pre-training Data via Diversified File Selection.


### Environment and Data Installation
For environment, we provide the detailed environment in environment.txt file.  
For data, you can download SlimPajama-627B through following command:
```bash
cd /path/to/dataset  
git lfs install  
git clone https://huggingface.co/datasets/cerebras/SlimPajama-627B  
```
or try other sources  
```bash
git clone https://gitee.com/hf-datasets/SlimPajama-627B
```
### Data pre-processing
You should first tokenize the datasets and divide them into chunks:  
```bash
python ./TinyLLama/scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split validation --percentage 1.0  
python ./TinyLLama/scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama --tokenizer_path data/llama  --destination_path data/slim_star_combined --split train --percentage 1.0
```
In parallel, you need to extract text features:
```bash
cd ./DISF
python extract_feature.py  
```
Notably, you should run 10 times of this command and modify the path in the python file to extract all chunk files into features.  

### Data Selection
See ./DISF to select files via DISF  

### Pre-train  
See ./TinyLLama to pre-train model.  
