import json
import glob
import os
from pathlib import Path
import sys
from typing import List
import numpy as np
from tqdm import tqdm
from multiprocessing import Process, cpu_count
from datasets import load_from_disk
from pathlib import Path
# dataset = load_from_disk('./data/ChnSentiCorp')
# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

import lit_gpt.packed_dataset as packed_dataset
from lit_gpt import Tokenizer

import pandas as pd


def prepare_full(
    source_path: Path,
    tokenizer_path: Path,
    destination_path: Path,
    chunk_size: int,
    split: str="train",
    filenames_subset: List[str] = None,
    process_id: int = 0
) -> None:
    import zstandard as zstd

    destination_path.mkdir(parents=True, exist_ok=True)

    tokenizer = Tokenizer(tokenizer_path)

    # Use the provided filenames_subset or default to all filenames
    filenames = filenames_subset 
    
    if not filenames:
        raise RuntimeError(
            f"No files matching  found at {source_path}. \n"
            "Make sure you download the data..."
        )

    builder = packed_dataset.PackedDatasetBuilder(
        outdir=destination_path,
        prefix=f"{split}_starcoder_{process_id}",  # Use process_id to differentiate builders
        chunk_size=chunk_size,
        sep_token=tokenizer.bos_id,
        dtype="auto",
        vocab_size=tokenizer.vocab_size,
    )
    #print(source_path)
    ds=load_from_disk("/ailab/group/medai-share/syDu/code_qurate/final_1percent5")
    #print(ds)
    for data in tqdm(ds):
        #print(data["content"])
        #break
        content=data['content']
        text_ids = tokenizer.encode(content)
        builder.add_array(np.array(text_ids, dtype=builder.dtype))
        #for item in content:
        #    print(item)
        #break
        #print(content)
        # for text in contents:
        #     text_ids = tokenizer.encode(text)
        #     builder.add_array(np.array(text_ids, dtype=builder.dtype))
    #print(filenames)
    # for filepath in tqdm(filenames):
        
    #     flag=True
    #     while(flag):
    #         print("Processing:",filepath)
    #         try:
    #             contents = pd.read_parquet(filepath, engine='pyarrow')['content']
    #             flag=False
    #         except:
    #             print(f"Error reading {filepath}!!")
    #             flag=True
    #             #continue
    #     for text in contents:
    #         text_ids = tokenizer.encode(text)
    #         builder.add_array(np.array(text_ids, dtype=builder.dtype))




                #continue
        # for text in contents:
        #     text_ids = tokenizer.encode(text)
        #     builder.add_array(np.array(text_ids, dtype=builder.dtype))

    # we throw away the final corpus to avoid meaningless corpus filled with bos_ids, see https://github.com/jzhang38/TinyLlama/issues/83 for more details
    # builder.write_reminder()


def prepare(
    source_path: Path = Path("data/RedPajama-Data-1T-Sample"),
    tokenizer_path: Path = Path("checkpoints/lit-llama/tokenizer.model"),
    destination_path: Path = Path("data/red_pajama_sample"),
    chunk_size: int = 2049 * 1024,
    split: str="train",
    percentage: float = 1.0,
    filenames_subset: List[str] = None,
) -> None:
    import time
    #assert split == "train" #  starcoder only has train data
    filenames = glob.glob(os.path.join(source_path, "./*.arrow"), recursive=True)
    # only retrain subsets that follow the prefix in filenames_subset
    if filenames_subset:
        filenames = [f for f in filenames if any([prefix in f for prefix in filenames_subset])]
    filenames = filenames[:int(len(filenames) * percentage)]
    num_processes = 1
    chunked_filenames = np.array_split(filenames, num_processes)

    processes = []
    start_time = time.time()

    for i, subset in enumerate(chunked_filenames):
        p = Process(target=prepare_full, args=(source_path, tokenizer_path, destination_path, chunk_size, split, list(subset), i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Time taken: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    from jsonargparse import CLI
    CLI(prepare)
