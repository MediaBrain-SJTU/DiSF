#### Step 1) selecting file indexs based on our disf
```bash
bash ./select_disf.sh
```
#### Step 2) merge selected indexs in each process
```bash
python ./merge.py
```
#### Step 3) transform selected indexs to raw text
```bash
python ./id2text.py
```
#### Step 4) tokenize the raw text 
```bash
python ../TinyLlama/scripts/prepare_slimpajama.py --source_path selected_raw_text_path \
--tokenizer_path data/llama --destination_path data/slim_star_combined \
--split validation --percentage 1.0
```
