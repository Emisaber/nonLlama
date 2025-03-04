# Sample from a trained model
# almost the same as nanoGPT
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import LlamaConfig, NonLlama

# CONFIG ---------------------------------------------------
init_from = 'resume'
out_dir = 'out'
start = "\n" # can be '<|endoftext|>' or a prompt file path starts with "FILE:"
num_samples = 10
max_new_tokens = 500
temperature = 0.8
top_k = 200
seed = 486 # your lucky number
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
compile = True
exec(open('configurator.py').read()) # overide the default config
# END CONFIG ----------------------------------------------

# TORCH SETTING -------------------------------------------
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
device_type = 'cuda' if 'cuda' in device else 'cpu'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)
# END TORCH SETTTING --------------------------------------

# INIT MODEL ----------------------------------------------
if init_from == 'resume':
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    
    llamaconf = LlamaConfig(**checkpoint['model_args'])
    model = NonLlama(llamaconf)
    
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    
    model.load_state_dict(state_dict=state_dict)

model.eval()
model.to(device)

if compile:
    model = torch.compile(model)
# END INIT MODEL ------------------------------------------

# TOKENIZER -----------------------------------------------
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']:
    meta_path = os.path.join(out_dir, checkpoint['config']['dataset'], 'meta.pkl')
    print(meta_path)
    load_meta = os.path.exists(meta_path)
    
if load_meta:
    print(f"Loading meta from {meta_path} ...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # load tokenizer from meta    
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long, device=device)[None, ...]
    decode = lambda s: ''.join([itos[t] for t in s])
    
else:
    print("No meta.pkl found, using GPT-2 tokenizer ...")
    enc = tiktoken.get_encoding('gpt2')
    encode = lambda s: torch.tensor(enc.encode(s, allowed_special={"<endoftext>"}))[None, ...]
    decode = lambda s: enc.decode(s)

if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
# END TOKENIZER ------------------------------------------

# GENERATE
input_seq = encode(start)

samples = []
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(input_seq, max_new_tokens, temperature=temperature)
            y = decode(y[0].tolist())
            samples.append(y)
            print("="*50)
            print(y)
            print("="*50)
            

# save generations
samples = '\n\n'.join(samples)
output_path = os.path.join(out_dir, 'samples.txt')
with open(output_path, 'w', encoding='utf-8') as f:
    f.write(samples)






















