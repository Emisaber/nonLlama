import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset
from huggingface_hub import HfApi
api = HfApi(token = "your_token")

#TODO use sentencepiece instead
# data prepare program almost copy from nanoGPT
num_proc = 16 # cpu cores//2, used in map()
num_proc_load_data = num_proc # number of workers when loading dataset

enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    
    dataset = load_dataset("fosaber/wikipedia-en-deduped-minhash", num_proc=num_proc_load_data)
    
    # remove useless columns
    dataset["train"] = dataset["train"].remove_columns(["id", "url", "title"])
    
    split_dataset = dataset["train"].train_test_split(test_size=0.0005, seed=2357, shuffle=True)
    split_dataset["val"] = split_dataset.pop('test')
    
    def process(example):
        ids = enc.encode_ordinary(example['text'])
        ids.append(enc.eot_token)
        
        out = {'ids': ids, 'len': len(ids)}
        return out
    
    tokenized = split_dataset.map(
        process,
        remove_columns=['text'],
        desc="tokenizing the data",
        num_proc=num_proc
    )
    
    for split, dset in tokenized.items():
        data_len = np.sum(dset['len'], dtype=np.uint64)
        # current dir
        filename = os.path.join(os.path.dirname(__file__), f'{split}.bin') # merge into a bin file
        # 50256 is smaller than 2**16
        data = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(data_len))
        total_batches = 1024 # write in batches
        
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
            data_batch = np.concatenate(batch['ids'])
            data[idx : idx + len(data_batch)] = data_batch
            idx = idx + len(data_batch)
        
        data.flush()
    api.upload_large_folder(
        repo_id = "wikipedia-en-deduped-minhash-bin",
        repo_type = "dataset",
        folder_path = os.path.dirname(__file__),
    )