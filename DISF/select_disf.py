import numpy as np
from torch.nn.functional import normalize
import numpy as np
from datasets import load_dataset
from torch.utils.data import IterableDataset, DataLoader
import time
from create_datasets import Dataset_jsonlzst,
import torch
from tqdm import tqdm
import random
import argparse
from multiprocessing import Process, cpu_count
from tqdm import tqdm

# Filename for SlimPajama
slimpajama_sets = {
    "chunk1": 59001760,
    "chunk2": 58991780,
    "chunk3": 59065701,
    "chunk4": 59045743,
    "chunk5": 59211340,
    "chunk6": 59031700,
    "chunk7": 58941880,
    "chunk8": 59097501,
    "chunk9": 59081600,
    "chunk10": 59001760,
}

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def hunger_select(feature,top_k=32,batch_size=2048,device_id=0):
    gpu="cuda:"+str(device_id)
    #print(gpu)
    feature=torch.tensor(feature).to(gpu)
    #print(feature.shape)
    selected_indexs=[]
    for i in range(top_k):
        #print(i)
        min_corre=10000
        new_index=-1
        for j in range(batch_size):
            #print(j)
            if j in selected_indexs:
                continue
            if (i==0):
                    new_index=random.randint(0,batch_size-1) 
                    break
            else:
                #print(feature[j:j+1].shape)
                feat=torch.cat((selected_features,feature[j:j+1]),dim=0)
                corre=cal_corre(feat)
                if corre<min_corre:
                    min_corre=corre
                    new_index=j

        if (i==0):
            selected_indexs.append(new_index)
            selected_features=feature[new_index:new_index+1]

        else:
            selected_indexs.append(new_index)
            selected_features=torch.cat((selected_features,feature[new_index:new_index+1]),dim=0)
    #print(selected_indexs)
    return selected_indexs,selected_features.to("cpu")

def cal_corre(feat_mat):

    batch_size=feat_mat.shape[0]
    eps=1e-8
    feat_mat = feat_mat - feat_mat.mean(dim=0, keepdim=True)
    feat_mat = feat_mat / torch.sqrt(eps + feat_mat.var(dim=0, keepdim=True))
    corre_mat=torch.matmul(feat_mat.t(), feat_mat)/(batch_size-1)
    corre=((off_diagonal(corre_mat).pow(2)).sum())

    return corre/10000

def select(
    feature,
    start,
    end,
    batch_size=512,
    top_k=8,
    n_devices=8,
    process_id=0,
    save_dir="./selected/contriever",
    file_name="chunk1"):
    num_iter=int((end-start+1)/batch_size)

    feat_local=feature
    device_id=process_id%n_devices
    for iter_idx in tqdm(range(num_iter)):
        
    
        feature_batch=feature[iter_idx*batch_size+start:(iter_idx+1)*batch_size+start]
        idx_prefix=iter_idx*batch_size
        idxs,feats=hunger_select(feature_batch,top_k=top_k,batch_size=batch_size,device_id=device_id)
        if iter_idx==0:
            
            selected_indexs=np.array(idxs)+start
            #selected_features=feats
        else:
            idxs=np.array(idxs)+idx_prefix+start
            selected_indexs=np.append(selected_indexs,idxs)
            #selected_features=torch.cat((selected_features,feats),dim=0)

    save_path=save_dir+"/"+file_name+"_selected_"+str(process_id)+".npy"
    np.save(save_path,selected_indexs)
    


def prepare(
    file_name= "chun1",
    file_dir="./data/contriever",
    save_dir="./selected/contriever/10B",
    scale=2,
    batchsize=512,
    top_k=8,
    feature_dim=768,
    n_devices=8,
) -> None:
    import time

    
    batch_size=int(batchsize*scale)

    file_path=file_dir+"/"+file_name+".bin"

    num_processes = cpu_count() 

    num_data=slimpajama_sets[file_name]
    top_k=int(8*scale)
    
    process_data=int(num_data/num_processes)

    feature=np.memmap(file_path, dtype='float32', mode='r', shape=(num_data,feature_dim))


    processes = []
    start_time = time.time()

    for i in range(num_processes):
        start=i*process_data
        end=(i+1)*process_data-1
        p = Process(target=select, args=(feature, start,end,batch_size,top_k,n_devices,i,save_dir,file_name))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='命令行参数')

    
    parser.add_argument('--file_name',  default="chunk1")
    parser.add_argument('--file_dir', default="your feature path")
    parser.add_argument('--save_dir',  default="./selected/contriever/16_1024")
    parser.add_argument('--scale',  default=2)
    parser.add_argument('--batchsize',default=512)
    parser.add_argument('--top_k', default=8)
    parser.add_argument('--feature_dim', default=768)
    parser.add_argument('--n_devices',  default=8)
    args = parser.parse_args()
    prepare(file_name= args.file_name,
    file_dir=args.file_dir,
    save_dir=args.save_dir,
    scale=args.scale,
    batchsize=args.batchsize,
    top_k=args.top_k,
    feature_dim=args.feature_dim,
    n_devices=args.n_devices,)
