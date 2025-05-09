import numpy as np
import torch

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def cal_egin(feat_mat):

    batch_size=feat_mat.shape[0]
    eps=1e-8
    feat_mat = feat_mat - feat_mat.mean(dim=0, keepdim=True)
    feat_mat = feat_mat / torch.sqrt(eps + feat_mat.var(dim=0, keepdim=True))
    corre_mat=torch.matmul(feat_mat.t(), feat_mat)/(batch_size-1)
    eigenvalue, _ = np.linalg.eig(corre_mat)
    eigenvalue=np.sort(eigenvalue)[::-1]
    ratio_list=[]
    for i in range(10):
        topk=(i+1)*10
        topk_egin=eigenvalue[0:topk]
        ratio=np.sum(topk_egin)/np.sum(eigenvalue)
        ratio_list.append(ratio)

    return ratio_list

yourpath=""  # selected feature path
batch_size=128  # number of samples used for calculating dominance score
feat_dim=768 # dimension of feature
selected_feature = torch.tensor(np.memmap(yourpath, dtype='float32', mode='r', shape=(batch_size,feat_dim)))

dominance_score=cal_egin(selected_feature)
print("dominance_score:",dominance_score)
