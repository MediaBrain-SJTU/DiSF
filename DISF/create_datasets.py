from torch.utils.data import IterableDataset, DataLoader
import random
import glob
import io
import zstandard as zstd
import json
import os
import jsonlines
from tqdm import tqdm
import pandas as pd
class Dataset_jsonlzst(IterableDataset):
    def __init__(self, filepath):
        filenames = sorted(glob.glob(filepath))

        self.filenames = filenames
    def __iter__(self):
        for filename in tqdm(self.filenames):
            with open(filename, 'rb') as file:
                decompressor = zstd.ZstdDecompressor()
                stream_reader = decompressor.stream_reader(file)
                stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")
                for line in stream:
                    yield json.loads(line)
    def cal_nums(self,index):
        filename=self.filenames[index]
        num=0
        with open(filename, 'rb') as file:
            decompressor = zstd.ZstdDecompressor()
            stream_reader = decompressor.stream_reader(file)
            stream = io.TextIOWrapper(stream_reader, encoding = "utf-8")
            for line in stream:
                num+=1
        print(num)

class Dataset_star(IterableDataset):
    def __init__(self, filepath):
        self.filepath=filepath
        filenames = sorted(glob.glob(os.path.join(filepath, "*/*.parquet"), recursive=True))

        self.filenames = filenames
    def __iter__(self):
        for filename in tqdm(self.filenames):
            contents = pd.read_parquet(filepath, engine='pyarrow')['content']
            for text in contents:
                yield text

    def cal_nums(self,index):
        
        filename=self.filenames[index]
        num=0
        contents = pd.read_parquet(filename, engine='pyarrow')['content']
        return len(contents)

    def cal_nums_all(self):
        num=0
        for filename in tqdm(self.filenames):
            contents = pd.read_parquet(self.filepath, engine='pyarrow')['content']
            del contents
            num+=len(contents)

        return num
if __name__=="__main__":
    data=Dataset_star("yourdatapath")
    print(data.cal_nums_all())
