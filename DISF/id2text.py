import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from create_datasets import Dataset_jsonlzst
import jsonlines

batch_size=1024
top_k=256
total_path="./total_selected/contriever/16_1024/chunk1.npy"
save_meta_path="./final_selected/contriever/16_1024/chunk1_meta.jsonl"
save_text_path="./final_selected/contriever/16_1024/chunk1_text.jsonl"
dataset_path='your data path/SlimPajama-627B/train/chunk1/*'
#selected_files=sorted(glob.glob(total_path))
flag="text"
selected_index=list(np.load(total_path))

dataset=Dataset_jsonlzst(dataset_path)
dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)
idx=0

saved_data={}
saved_data["text"]=[]
saved_data["meta"]={}
saved_data["meta"]["redpajama_set_name"]=[]
file_idx=0
count_idx=0

num=0
if flag=="meta":
    path=save_meta_path
else:
    path=save_text_path
overlist=[]
with jsonlines.open(path,"w") as w:

    for batch_data in dataloader:
        text_index=selected_index[idx*top_k:(idx+1)*top_k]
        if overlist:

            text_index.extend(overlist)

            overlist=[]

        for item in text_index:
            
            if(item-batch_size*idx>=batch_size):
                overlist.append(item)
                continue
            item=item-batch_size*idx
            
            selected_batch_domain=batch_data["meta"]["redpajama_set_name"][item]
            # if selected_batch_domain=="RedPajamaGithub":
            #     continue
            selected_batch_text=batch_data["text"][item]
            if flag=="meta":
                w.write(selected_batch_domain)
            else:
                w.write(selected_batch_text)

        idx+=1


