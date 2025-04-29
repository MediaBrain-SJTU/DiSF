import numpy as np
import glob
import jsonlines
from tqdm import tqdm
path="./contriever/16_1024/chunk1_selected_*.npy"  # your selected file path in disf
save_path="./total_selected/contriever/512_1024/chunk1.npy" # merged file path
files_name=sorted(glob.glob(path))
idx=0
for file in tqdm(files_name):
    if idx==0:
        data=np.load(file)
        
    else:
        data=np.append(data,np.load(file))
    idx+=1

data=np.sort(data)
np.save(save_path,data)
print(path,save_path)
