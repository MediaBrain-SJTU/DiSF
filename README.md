<p align="center" width="100%">
</p>

<div id="top" align="center">

(ICLR Oral, DiSF) Combatting Dimensional Collapse in LLM Pre-training Data via Diversified File Selection.
-----------------------------
<img src="https://img.shields.io/badge/Version-1.0.0-blue.svg" alt="Version"> 
<img src="https://img.shields.io/badge/License-Apache_2.0-green.svg" alt="License">

<h4> |<a href="https://arxiv.org/pdf/2504.20644"> 📑 Paper </a> |
<a href="https://github.com/MediaBrain-SJTU/DISF"> 🐱 Github Repo </a> |
</h4>

<!-- **Authors:** -->

_**Ziqing Fan<sup>1,2 </sup>, Siyuan Du<sup>2,3 </sup>, Shengchao Hu<sup>1,2</sup>, Pingjie Wang<sup>1,2</sup>, Li Shen<sup>4</sup>, Ya Zhang<sup>1,2</sup>, Dacheng Tao<sup>5</sup>, Yanfeng Wang<sup>1,2</sup>**_


<!-- **Affiliations:** -->


_<sup>1</sup> Shanghai Jiao Tong University,
<sup>2</sup> Shanghai AI Laboratory,
<sup>3</sup> Fudan University,
<sup>4</sup> Shenzhen Campus of Sun Yat-sen University,
<sup>5</sup> Nanyang Technological University,_

</div>

### To do list
visualization code release  
other baseline code release  
improvement on recent code  

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
python ./TinyLLama/scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama \
--tokenizer_path data/llama  --destination_path data/slim_star_combined \
--split validation --percentage 1.0  
python ./TinyLLama/scripts/prepare_slimpajama.py --source_path /path/to/SlimPajama \
--tokenizer_path data/llama  --destination_path data/slim_star_combined \
--split train --percentage 1.0
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
