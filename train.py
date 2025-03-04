# The script is aim at training on 2 A800 GPU
# run with DDP, 2 gpu on 1 node
# That's a lot of money, hope someday I could be rich
# almost the same as nanoGPT, just add some notes(and modify some number)

import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torch.utils.data.distributed import DistributedSampler

from model import LlamaConfig, NonLlama
from utils import PreTrainDataset, ChunkedRandomSampler, DistributedChunkedSampler, get_random_batch, get_lr_cos
# excellent import format !!


# CONFIG ---------------------------------------------------------
# Default: train a nonLlama (65M) 
# Dataset: wikipedia-en-deduped-minhash  https://huggingface.co/datasets/fosaber/wikipedia-en-deduped-minhash/tree/main

out_dir = 'out'
eval_interval = 100 # evaluate once per eval_interval 
log_interval = 1
eval_iters = 200  # how many batch when evaluate ?
eval_only = False
always_save_checkpoint = False
init_from = 'scratch'
num_proc_load_data = 2

# wandb logging
wandb_log = True
wandb_project = 'wikipedia-en'
wandb_run_name = 'wiki-nonllama'

# data
dataset = 'wikipedia-en-deduped-minhash'
gradient_accumulation_steps = 20 * 2 # only 2 gpu :(, simulate large batch size
batch_size = 24

# model
block_size = 1024
vocab_size = 50304
n_embd = 768
n_head = 16
n_layer = 10
n_kv_head = 8
dropout = 0.0 # 0 when pre-training, fine-tuning try 0.1+ 
bias = False # no bias inside Linear layers

# optimizer
peak_learning_rate = 6e-4 
max_iters = 300000 # need adjusted
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients

# learning rate
decay_lr = True
warmup_iters = 2000 # need adjusted, how many iterations to warm up for
lr_decay_iters = 300000 # per Chinchilla
min_lr = 6e-5

# DDP network protocol
backend = 'nccl'

# system
gpu = "vGPU-32GB"
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
print(f"data type: {dtype}")
compile = True

# change global config
# may not elegant enough but I like it    orz
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read())
config = {k: globals()[k] for k in config_keys}
# for i, v in config.items():
#     print(f"Set {i}: {v}")
# END CONFIG ------------------------------------------------------

# DDP CONFIG-------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1

if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    
    master_process = ddp_rank == 0  # rank 0 process will do logging, checkpointing etc.
    
    seed_offset = ddp_rank # each process gets a different seed
    
    assert gradient_accumulation_steps % ddp_world_size == 0 # divide the batch equally among each process
    gradient_accumulation_steps //= ddp_world_size

else:
    print("Using a single gpu/cpu")
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
# END DDP CONFIG -------------------------------------------------

# TRAINING CONFIG ------------------------------------------------
token_per_iters = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {token_per_iters:,}") # print in a format like 1,000,000

if master_process:
    os.makedirs(out_dir, exist_ok=True)

torch.manual_seed(486 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # used in autocast

ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.autocast(device_type=device_type, dtype=ptdtype)

current_iter_num = 0
best_val_loss = 1e9  # could be override if init_from = 'resume'
# END TRAINING CONFIG -------------------------------------------


# DATA CONFIG ---------------------------------------------------
data_dir = os.path.join('data', dataset)
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    assert "train_data_len" in meta and "val_data_len" in meta
    train_data_len, val_data_len = meta["train_data_len"], meta["val_data_len"]
    if "vocab_size" in meta:
        meta_vocab_size = meta['vocab_size']
        print(f"found vocab_size = {meta_vocab_size} in {meta_path}")
    

# END CONFIG ----------------------------------------------------


# MODEL INIT ----------------------------------------------------
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, n_kv_head=n_kv_head, 
                  vocab_size = vocab_size, block_size=block_size, 
                  bias=bias, dropout=dropout)

if init_from == 'scratch':
    print("Initializing from scratch")
    
    if meta_vocab_size is not None:
        print(f"Using vocab_size == meta vocab size founded in {meta_path}")
    else:
        print("Using GPT-2 vocab size")
    
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    

    llamaconf = LlamaConfig(**model_args)
    model = NonLlama(llamaconf)
    
elif init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    print(f"Resuming training from {ckpt_path}")
    
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    
    # Copy args from checkpoint
    for k in ['n_layer', 'n_head', 'n_head', 'n_embd', 'n_kv_head', 
              'vocab_size', 'block_size', 'bias', 'dropout']:
        model_args[k] = checkpoint_model_args[k]
    
    llamaconf = LlamaConfig(model_args)
    model = NonLlama(llamaconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict)
    current_iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# END INIT MODEL -------------------------------------------------

# OPTIMIZER ------------------------------------------------------
scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))
optimizer = model.configure_optimizers(weight_decay=weight_decay, learning_rate=peak_learning_rate,
                                       betas=(beta1, beta2), device_type=device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
# END OPTIMIZER --------------------------------------------------

checkpoint = None # free up memory

# WRAP MODEL -----------------------------------------------------
model.to(device)
if compile:
    print("Compiling the model....")
    uoptimized_model = model
    model = torch.compile(model)

if ddp:
    print("Using DDP...")
    model = DDP(model, device_ids=[ddp_local_rank])

# END WRAP MODEL -------------------------------------------------

# ESTIMATE LOSS --------------------------------------------------
#TODO check for correctness
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_random_batch(data_dir, split, block_size, batch_size, device)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
# END ESTIMAT LOSS -----------------------------------------------

# WANDB LOGGING --------------------------------------------------
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
# END WANDB LOGGING ----------------------------------------------

# TRANING LOOP ---------------------------------------------------
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model
running_mfu = -1.0

def eval():
    lr = get_lr_cos(it=current_iter_num, warmup_iters=warmup_iters,
                    lr_decay_iters=lr_decay_iters, max_lr=peak_learning_rate, min_lr=min_lr)
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    losses = estimate_loss()
    print(f"step {current_iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    if wandb_log:
        wandb.log({
            "iter": current_iter_num,
            "train/loss": losses['train'],
            "val/loss": losses['val'],
            "lr": lr,
            "mfu": running_mfu * 100,
        })
    global best_val_loss
    if losses['val'] < best_val_loss or always_save_checkpoint:
        best_val_loss = losses['val']
        if current_iter_num > 0:
            checkpoint = {
                "model": raw_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "model_args": model_args,
                "iter_num": current_iter_num,
                "best_val_loss": best_val_loss,
                "config": config,
            }
            
            print(f"saving checkpoint to {out_dir}")
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))


# START TRAINING/EVALUATION
if __name__ == "__main__":

    pretrain_train_dataset = PreTrainDataset(data_dir=data_dir, split="train", data_len=val_data_len, block_size=block_size)
    
    if ddp:
        # sampler = DistributedSampler(pretrain_train_dataset)
        sampler = DistributedChunkedSampler(pretrain_train_dataset, replacement=True)
        train_dataloader = DataLoader(dataset=pretrain_train_dataset, batch_size=batch_size,
                                      shuffle=False, sampler=sampler, num_workers=num_proc_load_data, pin_memory=True)
    else:
        sampler = ChunkedRandomSampler(pretrain_train_dataset, replacement=True)
        train_dataloader = DataLoader(dataset=pretrain_train_dataset, batch_size=batch_size, sampler=sampler,
                                      shuffle=False, num_workers=num_proc_load_data, pin_memory=True)

    if eval_only and master_process:
        eval()

    if not eval_only:
        
        # Poor man's iteration
        while True:
            
            if ddp:
                sampler.set_epoch(local_iter_num)
            data = iter(train_dataloader)
            
            while True:
                #evaluate the loss on train/val sets and save checkpoints
                if current_iter_num % eval_interval == 0 and master_process:
                    print("evaluating .....")
                    eval()
                
                try:
                    for micro_step in range(gradient_accumulation_steps):
                        micro_batch = next(data)
                        X, Y = micro_batch
                        X, Y = X.to(device, non_blocking=True), Y.to(device, non_blocking=True)
                        if ddp:
                            # sync gradients at the last micro step
                            model.require_backward_grad_sync = (micro_step == (gradient_accumulation_steps - 1))
                        with ctx:
                            logits, loss = model(X, Y)
                            loss = loss / gradient_accumulation_steps
                        scaler.scale(loss).backward()
                    
                    if grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    scaler.step(optimizer)
                    scaler.update() # update scaler factor
                    optimizer.zero_grad(set_to_none=True)
    
                    t1 = time.time()
                    dt = t1 - t0
                    t0 = t1
                    if current_iter_num % log_interval == 0 and master_process:
                        lossf = loss.item() * gradient_accumulation_steps
                        if local_iter_num >= 5:
                            mfu = raw_model.estimate_mfu(batch_size*gradient_accumulation_steps, dt, gpu)
                            running_mfu = mfu if running_mfu == -1.0 else running_mfu * 0.9 + mfu * 0.1
                        print(f"iter {current_iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
                    current_iter_num += 1
                    local_iter_num += 1
                    if current_iter_num > max_iters:
                        raise StopIteration
                    
                except StopIteration:
                    break
            if current_iter_num > max_iters:
                break
        if master_process:
            eval()
                
    if ddp:
        destroy_process_group()