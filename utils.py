import os
import math
import torch
import numpy as np
import gc
import torch.distributed as dist
from torch.utils.data import Dataset, Sampler


# Dataset
class PreTrainDataset(Dataset):
    def __init__(self, data_dir, split, data_len, block_size=1024):
        assert split in ["train", "val"]
        self.data_dir = data_dir
        self.split = split
        self.block_size = block_size
        self.len = data_len
        
    def __len__(self):
        return self.len - self.block_size
    
    def __getitem__(self, index):
        data = np.memmap(os.path.join(self.data_dir, f"{self.split}.bin"),
                        dtype=np.uint16, mode='r')
        x = torch.from_numpy(np.array(data[index : index + self.block_size]).astype(np.int64))
        y = torch.from_numpy(np.array(data[index + 1 : index + self.block_size + 1]).astype(np.int64))

        
        return x, y

class ChunkedRandomSampler(Sampler[int]):
    def __init__(
        self, 
        data_source, 
        chunk_size = 100000000, 
        num_samples = None, 
        generator = None,
        replacement = False
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.data_source = data_source
        self.generator = generator
        self._num_samples = num_samples
        self.replacement = replacement
        
    @property
    def num_samples(self):
        if self._num_samples is None:
            return len(self.data_source) # dataset may change, return the len instead of a fixed num
        else:
            return self._num_samples
    
    def __len__(self):
        return self.num_samples
    
    
    def __iter__(self):
        n = len(self.data_source)
        chunk_size = self.chunk_size
        num_samples = self.num_samples
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator
            
        if self.replacement:
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=generator
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=generator,
            ).tolist()
        else:    
            for start in range(0, num_samples, chunk_size):
                end = min(start+chunk_size, num_samples) #if num_samples is less than n, drop last actually
                yield from (torch.randperm(end-start, generator=generator) + start).tolist()


class DistributedChunkedSampler(Sampler):
    def __init__(
        self,
        dataset,
        num_replicas = None,
        rank = None,
        chunk_size = 100000000, #GPU
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
        replacement: bool = False
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.chunk_size = chunk_size
        self.replacement = replacement

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  
            )

        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed
        
    def __len__(self):
        return self.num_samples

    def __iter__(self):
        total_chunk_size = self.chunk_size
        max_len = min(self.total_size, len(self.dataset))
        padding_size = self.total_size - len(self.dataset)

        if self.replacement:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            n = len(self.dataset)
            for _ in range(self.num_samples // 32):
                yield from torch.randint(
                    high=n, size=(32,), dtype=torch.int64, generator=g
                ).tolist()
            yield from torch.randint(
                high=n,
                size=(self.num_samples % 32,),
                dtype=torch.int64,
                generator=g,
            ).tolist()
        else:
            if self.shuffle:
                g = torch.Generator()
                g.manual_seed(self.seed + self.epoch)
                for idx in range(0, max_len, total_chunk_size):
                    end = min(max_len, idx+total_chunk_size)
                    indices = (torch.randperm(end, generator=g)+idx).tolist()
                    if not self.drop_last and end == max_len:
                        indices += (torch.randperm(padding_size, generator=g)).tolist()
                    yield from indices[self.rank : self.total_size : self.num_replicas]
            else:
                for idx in range(0, max_len, total_chunk_size):
                    end = min(max_len, idx+total_chunk_size)
                    indices = list(range(idx, end))
                    if not self.drop_last and end == max_len:
                        indices +=  list(range(padding_size))
                    yield from indices[self.rank : self.total_size : self.num_replicas]

    def set_epoch(self, epoch):
        self.epoch = epoch       


def get_random_batch(data_dir, split, block_size, batch_size, device):

    data = np.memmap(os.path.join(data_dir, f'{split}.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    
    if 'cuda' in device:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)

    return x, y


def get_lr_cos(it, warmup_iters, lr_decay_iters, max_lr, min_lr):
    
    if it < warmup_iters:
        return max_lr * (it + 1) / (warmup_iters + 1)
    
    if it > lr_decay_iters:
        return min_lr
    
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # 1 -> 0
    
    return min_lr + coeff * (max_lr - min_lr)
    



