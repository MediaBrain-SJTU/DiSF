import torch
import numpy as np
from torch.utils.data import IterableDataset, DataLoader
from create_datasets import Dataset_jsonlzst
import torch
from transformers import AutoTokenizer, AutoModel

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def get_feature(emb_array,dataloader,model_name="contriever",device=torch.device("cuda:0")):
    # initialize feature extractor and tokenizer, you may use other extractor
    if model_name=="contriever":
        tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
        model = AutoModel.from_pretrained('facebook/contriever')
    model.to(device)
    
    idxx=0

    # extract features and save them into 'emb_array'
    for train_data in dataloader:
        batch_size=len(train_data)
        text_data=train_data["text"]
        inputs = tokenizer(text_data, padding=True, truncation=True, return_tensors='pt')
        inputs={key:inputs[key].to(device) for key in inputs}
        with torch.no_grad():
            outputs = model(**inputs)
            embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
            emb_array[idxx:idxx+batch_size] = embeddings.to("cpu")
        idxx+=batch_size
    return 0

statistic={
  "chunk1": 59001760
  "chunk2": 58991780
  "chunk3": 59065701
  "chunk4": 59045743
  "chunk5": 59211340
  "chunk6": 59031700
  "chunk7": 58941880
  "chunk8": 59097501
  "chunk9": 59081600
  "chunk10": 59001760
}

# you should run 10 times, since there are 10 chunk file in SlimPajama.  
batch_size=512
feature_dim=768
chunk_file="chunk1"
embed_model_name="contriever"
datapath="your raw download path"
emb_memory_loc="./data/contriever/"+chunk_file +".bin"
data_size=statistic[chunk_file]
dataset=Dataset_jsonlzst('datapath/train/chunk1/*')
device=torch.device("cuda:0")

# do not use shuffle!! Using shuffle will lose the order of files and you will not know the exact file through id after selection
# If you define each setence with a unique id, you can try to use shuffle
dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size, num_workers=0)
emb_array = np.memmap(emb_memory_loc, dtype='float32', mode='w+', shape=(data_total, 768))

# you can accelerate this procedures by running more this command on your device with different chunk_file.
get_feature(emb_array=emb_array,dataloader=dataloader,model_name=embed_model_name,device=device)
