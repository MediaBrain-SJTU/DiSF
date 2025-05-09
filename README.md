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
visualization code release  **Done**  
other baseline code release  
improvement on recent code  
extracted data, and model release  
evaluation  
### 1. Environment

For environment, we provide the following command to construct based on Tinyllama repo(https://github.com/jzhang38/TinyLlama/blob/main/PRETRAIN.md):
```bash
pip install --index-url https://download.pytorch.org/whl/nightly/cu118 --pre 'torch>=2.1.0dev'
pip uninstall ninja -y && pip install ninja -U
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install
cd csrc/rotary && pip install .
cd ../layer_norm && pip install .
cd ../xentropy && pip install .
cd ../.. && rm -rf flash-attention
pip install -r ./requirements.txt tokenizers sentencepiece
```
As for detailed environments used in our experiments, we provide them in environment.txt file.  

### 2. Data Prepare

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
### 3. Data pre-processing
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

### 4. Data Selection
See ./DISF to select files via DISF  

### 5. Pre-train  
See ./TinyLLama to pre-train model.  

### 6. Model Version Transfer and Evaluation  
to be continued  

### 7. Visualization and Verification of Dimensional Collapse  
1. You should first extract features of your selected files (**See Data pre-processing part**)  
2. Before using following codes to visualize dimensional collapse, you should define your data path and fig save path in **./Visual&verify/collapse.py**  
```bash
cd ./Visual&verify/
python collapse.py  
```
3. Similarily, before using following codes to calculate dominance score, you should define your data path in **./Visual&verify/dominance_score.py**  
```bash
cd ./Visual&verify/
python dominance_score.py  
```


## Citation
If you find this work is relevant with your research or applications, please feel free to cite our work!
```
@article{fan2025combatting,
  title={Combatting Dimensional Collapse in LLM Pre-Training Data via Diversified File Selection},
  author={Fan, Ziqing and Du, Siyuan and Hu, Shengchao and Wang, Pingjie and Shen, Li and Zhang, Ya and Tao, Dacheng and Wang, Yanfeng},
  journal={arXiv preprint arXiv:2504.20644},
  year={2025}
}
```
