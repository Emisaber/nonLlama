
## `model.py`


```python
import math
import inspect
from dataclasses import dataclass
```
- dataclass作为decorator，用于自动为类添加`__init__()` `__repr__()`, `__eq__()`，`__ne__()` 等方法   
- `inspect` 用于访问优化器的参数

#### LayerNorm 
为了能将bias设为None   
有时间的话补一下LayerNorm？   
```python
class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)
```
> 提供了一个None的接口，应该是为了方便后面的参数计算


#### Attention block  

```python
class CausalSelfAttention(nn.Module):
```


**初始化**  
```python
def __init__(self, config):
	super().__init__()
	assert config.n_embd % config.n_head == 0 # 如果不能整除中断
	# key, query, value projections for all heads, but in a batch
	self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
	# output projection
	self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
	# regularization
	self.attn_dropout = nn.Dropout(config.dropout)
	self.resid_dropout = nn.Dropout(config.dropout)
	self.n_head = config.n_head
	self.n_embd = config.n_embd
	self.dropout = config.dropout
	# flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
	self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
	if not self.flash:
		print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
		# causal mask to ensure that attention is only applied to the left in the input sequence
		self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
```
- 初始化QKV权重矩阵(在这里将它们作为一个batch得到)
- 投影矩阵用于残差连接前进行投影，定义注意力的dropout，残差连接的dropout，定义注意力头数，embedding长度，dropout概率  
- 通过`hasattr` 判断当前torch是否有flash attention，没有的话与原理视频实现一致
	- 这里的`view`应该不加也行

**前传**  
```python
def forward(self, x):
	B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

	# calculate query, key, values for all heads in batch and move head forward to be the batch dim
	q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
	k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
	q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
	v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

	# causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
	if self.flash:
		# efficient attention using Flash Attention CUDA kernels
		y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
	else:
		# manual implementation of attention
		att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
		att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
		att = F.softmax(att, dim=-1)
		att = self.attn_dropout(att)
		y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
	y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

	# output projection
	y = self.resid_dropout(self.c_proj(y))
	return y
```
- 将权重矩阵切成三份
- 与原理视频不同的在于区分了不同注意力头 `B, nh, T, hs`
- flash attention 或者 slow attention 输出为 `B, nh, T, T`
	- 注意力矩阵的最后一维为时间
- `y = y.transpose(1,2).contiguous().view(B, T, C)`  将不同注意力头的输出拼起来
- 最后进行投影并dropout


#### MLP

```python
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
```
- 使用与transformer一致的4倍中间状态
- 将原理视频中的ReLU换成GELU

#### Block

Transformer block  
```python
class Block(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
```

#### GPT

```python
@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
```
- dataclass自动添加了基本的类方法，用于得到统一的GPT config


```python
class GPT(nn.Module):
```

```python
def __init__(self, config):
	super().__init__()
	assert config.vocab_size is not None
	assert config.block_size is not None
	self.config = config

	self.transformer = nn.ModuleDict(dict(
		wte = nn.Embedding(config.vocab_size, config.n_embd),
		wpe = nn.Embedding(config.block_size, config.n_embd),
		drop = nn.Dropout(config.dropout),
		h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
		ln_f = LayerNorm(config.n_embd, bias=config.bias),
	))
	self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
	# with weight tying when using torch.compile() some warnings get generated:
	# "UserWarning: functional_call was passed multiple values for tied weights.
	# This behavior is deprecated and will be an error in future versions"
	# not 100% sure what this is, so far seems to be harmless. TODO investigate
	self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying

	# init all weights
	self.apply(self._init_weights)
	# apply special scaled init to the residual projections, per GPT-2 paper
	for pn, p in self.named_parameters():
		if pn.endswith('c_proj.weight'):
			torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

	# report number of parameters
	print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))
```
- Weight Tying
	- embedding 和 softmax层权重共享，能优化模型表现（Attention is all you need 采用了这个方法）
	- 代码中实现 `self.transformer.wte.weight = self.lm_head.weight`，但是有warning
	- [Why Weight Tying](https://emisaber.github.io/White_Box/Notes/Why-Weight-Tying)
- `self.apply(self._init_weights)`  
	- 对模型进行特殊的权重初始化 
	- maybe an empirical method

**获得总参数量**   
```python
def get_num_params(self, non_embedding=True):
	"""
	Return the number of parameters in the model.
	For non-embedding count (default), the position embeddings get subtracted.
	The token embeddings would too, except due to the parameter sharing these
	params are actually used as weights in the final layer, so we include them.
	"""
	n_params = sum(p.numel() for p in self.parameters())
	if non_embedding:
		n_params -= self.transformer.wpe.weight.numel()
	return n_params
```
- 计算参数量，`numel()` 返回tensor element的个数
- 默认不加算embedding的参数量
	- Transformer中应是没有position embedding 参数的，一开始的embedding的参数也应该不被包含，但是它和投影层共享，所以实际上也计算在内

**初始化权重**(**易于收敛**)
```python
def _init_weights(self, module):
	if isinstance(module, nn.Linear):
		torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
		if module.bias is not None:
			torch.nn.init.zeros_(module.bias)
	elif isinstance(module, nn.Embedding):
		torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
```
- 🤔 这个`std=0.02`应是实验结论


```python
def forward(self, idx, targets=None):
	device = idx.device
	b, t = idx.size()
	assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
	pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

	# forward the GPT model itself
	tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
	pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
	x = self.transformer.drop(tok_emb + pos_emb)
	for block in self.transformer.h:
		x = block(x)
	x = self.transformer.ln_f(x)

	if targets is not None:
		# if we are given some desired targets also calculate the loss
		logits = self.lm_head(x)
		loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
	else:
		# inference-time mini-optimization: only forward the lm_head on the very last position
		logits = self.lm_head(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
		loss = None

	return logits, loss
```
- 前传过程
	- token embedding + pos embedding
	- dropout
	- each block in transformer
	- layernorm at the end of transformer
	- return loss if traning else logits only

**裁剪上下文**  
```python
def crop_block_size(self, block_size):
	# model surgery to decrease the block size if necessary
	# e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
	# but want to use a smaller block size for some smaller, simpler model
	assert block_size <= self.config.block_size
	self.config.block_size = block_size
	self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
	for block in self.transformer.h:
		if hasattr(block.attn, 'bias'):
			block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]
```
- 如果需要更小的block size，对模型的参数进行裁剪

**从预训练模型加载**  
```python
@classmethod
def from_pretrained(cls, model_type, override_args=None):
	assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
	override_args = override_args or {} # default to empty dict
	# only dropout can be overridden see more notes below
	assert all(k == 'dropout' for k in override_args)
	from transformers import GPT2LMHeadModel
	print("loading weights from pretrained gpt: %s" % model_type)

	# n_layer, n_head and n_embd are determined from model_type
	config_args = {
		'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
		'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
		'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
		'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
	}[model_type]
	print("forcing vocab_size=50257, block_size=1024, bias=True")
	config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
	config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
	config_args['bias'] = True # always True for GPT model checkpoints
	# we can override the dropout rate, if desired
	if 'dropout' in override_args:
		print(f"overriding dropout rate to {override_args['dropout']}")
		config_args['dropout'] = override_args['dropout']
	# create a from-scratch initialized minGPT model
	config = GPTConfig(**config_args)
	model = GPT(config)
	sd = model.state_dict()
	sd_keys = sd.keys()
	sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param

	# init a huggingface/transformers model
	model_hf = GPT2LMHeadModel.from_pretrained(model_type)
	sd_hf = model_hf.state_dict()

	# copy while ensuring all of the parameters are aligned and match in names and shapes
	sd_keys_hf = sd_hf.keys()
	sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
	sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)
	transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
	# basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
	# this means that we have to transpose these weights when we import them
	assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"
	for k in sd_keys_hf:
		if any(k.endswith(w) for w in transposed):
			# special treatment for the Conv1D weights we need to transpose
			assert sd_hf[k].shape[::-1] == sd[k].shape
			with torch.no_grad():
				sd[k].copy_(sd_hf[k].t())
		else:
			# vanilla copy over the other parameters
			assert sd_hf[k].shape == sd[k].shape
			with torch.no_grad():
				sd[k].copy_(sd_hf[k])

	return model
```
- 把`from transformers import GPT2LMHeadModel` 写在函数内，可能是`from_pretrained`不一定会使用，这样比较节约内存和高效
- 通过list comprehension `sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')]` 丢弃 buffer
- GPT openai的实现使用`Conv1D` 进行投影 `c_attn`是注意力的权重，`c_proj`是注意力之后的embedding 投影权重，`mlp.c_fc` `mlp.c_proj` 是MLP的两个线性层
- `copy_()` inplace copy，将预训练权重中卷积的部分转化为线性层权重

**优化器设置**  
```python
def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
	# start with all of the candidate parameters
	param_dict = {pn: p for pn, p in self.named_parameters()}
	# filter out those that do not require grad
	param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
	# create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
	# i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
	decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
	nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
	optim_groups = [
		{'params': decay_params, 'weight_decay': weight_decay},
		{'params': nodecay_params, 'weight_decay': 0.0}
	]
	num_decay_params = sum(p.numel() for p in decay_params)
	num_nodecay_params = sum(p.numel() for p in nodecay_params)
	print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
	print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
	# Create AdamW optimizer and use the fused version if it is available
	fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
	use_fused = fused_available and device_type == 'cuda'
	extra_args = dict(fused=True) if use_fused else dict()
	optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
	print(f"using fused AdamW: {use_fused}")

	return optimizer
```
- 对parameters分组，只对权重参数进行正则化，不对偏置项进行正则化
	- 偏置项并不主要参与模型输出计算，只对结果进行平移/缩放，偏置项的值通常比较小，对偏置项进行正则化效果不明显，可能导致欠拟合(影响了灵活性)
	- 这个操作应该可以当成模板
- `fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters`
	- `inspect.signature` 可以查看`callable`(`Adamw` here)的参数
	- `fused`是对优化器的性能的一种优化
	- 这个也是模板

**估计MFU**   
```python
def estimate_mfu(self, fwdbwd_per_iter, dt):
	""" estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
	# first estimate the number of flops we do per iteration.
	# see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
	N = self.get_num_params()
	cfg = self.config
	L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
	flops_per_token = 6*N + 12*L*H*Q*T
	flops_per_fwdbwd = flops_per_token * T
	flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
	# express our flops throughput as ratio of A100 bfloat16 peak flops
	flops_achieved = flops_per_iter * (1.0/dt) # per second
	flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
	mfu = flops_achieved / flops_promised
	return mfu
```
- MFU: model flops utilization
- FLOPs计算(来自PaLM) $R = \frac{P}{6N+12LHQT}$ 其中 $6N$ 来自parameter，$12LHQT$ 来自attention的两次矩阵运算?，在这里计算反过来了
- `dt`是迭代一次的时间

**生成**    
```python
@torch.no_grad()
def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
	"""
	Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
	the sequence max_new_tokens times, feeding the predictions back into the model each time.
	Most likely you'll want to make sure to be in model.eval() mode of operation for this.
	"""
	for _ in range(max_new_tokens):
		# if the sequence context is growing too long we must crop it at block_size
		idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
		# forward the model to get the logits for the index in the sequence
		logits, _ = self(idx_cond)
		# pluck the logits at the final step and scale by desired temperature
		logits = logits[:, -1, :] / temperature
		# optionally crop the logits to only the top k options
		if top_k is not None:
			v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
			logits[logits < v[:, [-1]]] = -float('Inf')
		# apply softmax to convert logits to (normalized) probabilities
		probs = F.softmax(logits, dim=-1)
		# sample from the distribution
		idx_next = torch.multinomial(probs, num_samples=1)
		# append sampled index to the running sequence and continue
		idx = torch.cat((idx, idx_next), dim=1)

	return idx
```
- 限制输入的序列长度(对过长的截断)
- 使用`topk`进行generate



## `train.py`

training script   

#### hyperparameter 

```python
out_dir = 'out'
eval_interval = 2000
log_interval = 1
eval_iters = 200
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = True # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = False # disabled by default
wandb_project = 'owt'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'openwebtext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 12 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = True # use PyTorch 2.0 to compile the model to be faster
```
- DDP:  DistributedDataParallel   
- `wandb`: 是一个AI developer platform，提供一系列工具便利模型训练和可视化   
- `gradient_accumulation_steps`: 用于在小内存的情况下模拟大batch size，每一个minibatch的gradient会累积直到到达指定的步数   (如果使用gradient accumulation steps，batch size应该适当减小)    
- `weight_decay`:  weight_decay代表L2正则化的能力
- `beta1, beta2`:  是AdamW中的两个参数
- `decay_lr`, `warmup_iters`, `lr_decay_iters`, `min_lr` 用于调整学习率  
	- Chinchilla是谷歌scaling law论文的模型名字，不知道这里指的是什么  
- backend 是DDP的设置  
- `torch.compile`  是pytorch 2.0引入的新特性，通过即时编译技术显著提升模型训练和推理速度。 

```python
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
```
- `globals()` 返回当前模块全局命名空间的字典，`python`中每个文件会有自己的全局命名空间，存储了定义的所有全局变量
- `config_keys`遍历所有全局变量找到所有config
- `exec(open('configurator.py').read())` 读取`configurator.py`中的代码然后执行，修改全局命名空间中的全局变量
- 这个的实现(原话)不是很好，直接import虽然能执行代码但是不会修改当前的globals，如果用parser解析输入的参数，需要对每个config variable都写一次，确实也不是很好


#### DDP 

```python
ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    init_process_group(backend=backend)
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
    seed_offset = ddp_rank # each process gets a different seed
    # world_size number of processes will be training simultaneously, so we can scale
    # down the desired gradient accumulation iterations per process proportionally
    assert gradient_accumulation_steps % ddp_world_size == 0
    gradient_accumulation_steps //= ddp_world_size
else:
    # if not ddp, we are running on a single gpu, and one process
    master_process = True
    seed_offset = 0
    ddp_world_size = 1
    
tokens_per_iter = gradient_accumulation_steps * ddp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")
```
- 比较需要说明的应该是
- 为每个process设置一个seed
- `gradient_accumulation_steps` 被各个进程平分
- `token_per_iter`计算了一次迭代训练了多少token

```python
if master_process:
    os.makedirs(out_dir, exist_ok=True)
```
- master process 用于记录日志，保存checkpoints   

```python
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```
- `torch.autocast` 是pytorch的上下文管理器，用于自动混合精度训练(Automatic Mixed Precision, AMP)。在某些操作时使用低精度(float16)加速计算，减少内存占用，在某些计算使用float32避免数值不稳定
- `GradScaler` 当使用float16时，由于精度较低，可能导致梯度下溢(小数值被截断或为0)，需要使用`GradScaler`缩放损失值。指定`autocast`的dtype为float16时会自动使用
- `nullcontext()` 返回一个空的上下文管理器，它是上下文管理器但又什么都不做，用于展位或者简化代码，与`autocast`应是搭配出现（这应该可以当成板子）

#### Training

##### data loader

```python
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
    else:
        data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y
```
- 这仍然和之前一样是一个自己实现的data loader，随机获取一个批次的数据
- 由于数据过大，使用`memmap`存储数据
	- `memmap`不将数据直接存储在内存中，而是存储在disk中，通过lazy loading加载到内存中。此处的实现参考了[python - numpy memmap memory usage - want to iterate once - Stack Overflow](https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122)，每个批次的sample都创建了一次`memmap`，避免内存泄漏
- 如果能够使用`cuda`，则将数据放到GPU中。`pin_memory`将数据固定在内存中不会被操作系统置换出来，加上`non_blocking=True`实现了asynchronously转移到GPU

##### 初始化

```python
iter_num = 0
best_val_loss = 1e9
```
- 初始化迭代次数和最佳的loss

```python
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")
```
- 从数据集的`meta.pkl`如果有的话寻找`vocab_size`

**初始化模型**   
```python
# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    if meta_vocab_size is None:
        print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
    model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']
elif init_from.startswith('gpt2'):
    print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
    # initialize from OpenAI GPT-2 weights
    override_args = dict(dropout=dropout)
    model = GPT.from_pretrained(init_from, override_args)
    # read off the created config params, so we can store them into checkpoint correctly
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)
```
- `model_args` 作为`GPTconfig`的参数
- `scratch`为从头初始化模型，`resume`为从checkpoint恢复
- `checkpoint = torch.load(ckpt_path, map_location=device)` 中`map_location`将张量加载到`device`中
- 中间去掉不需要的前缀应该是试错得到的，需要留意一下(指抄一下)，这里为什么不直接从checkpoint加载呢？
- 最后如果用GPT2的权重进行初始化，只需要`model = GPT.from_pretrained(init_from, override_args)`，为了正确checkpoint，需要将模型config转移到字典`model_args`
- 按需修改一下`block_size`
- 将模型转移到device中

```python
# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))
```
- 初始化`scaler`，`scaler`作用见上文

**优化器**   
```python
# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory
```
- 调用`model`中实现的`configure_optimizers`得到optimizer
- 如果是恢复的需要恢复optimizer状态
- 释放checkpoint变量

**Compile**  
```python
# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0
```
- `torch.compile` 生成优化后的计算图，不原地修改模型，`unoptimized_model`这里应该是维护一个未优化的模型副本，虽然后面的代码没用到这个
- 应该是一种板子

```python
# wrap model into DDP container
if ddp:
    model = DDP(model, device_ids=[ddp_local_rank])
```
- 加上DDP容器

##### 功能函数

```python
# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
```
- 迭代`eval_iters`次，得到相应次数的batch来获得更精确的平均loss，这个可能会涉及到训练数据的大小计算，按理说要`eval_iters`乘以`batch`要小于总数据量比较好(换数据集需要注意)
- `ctx` 是上文定义的上下文管理器，如果有GPU的话就是`torch.cuda.amp.autocast`，用FP16加速计算

```python
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * (it + 1) / (warmup_iters + 1)
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)
```
- 当小于`warmup`步数时linear warmup，当大于learning rate decay步数时直接返回最小lr，中间使用cosine decay
- `+1` 避免初始学习率为0
- `dacay_ratio` 计算当前迭代次数占比，用于调整cosine的变化，`0.5`即1/2，限制学习率大小在0,1之间，从0到$\pi$，从1到-1

```python
# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
```
- 用`wandb`记录日志(得查一下autodl怎么用wandb可能)

##### Trainning loop

```python
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model.module if ddp else model # unwrap DDP container if needed
running_mfu = -1.0
```
- 进行简单的初始化

```python
while True:

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    # evaluate the loss on train/val sets and write checkpoints
    if iter_num % eval_interval == 0 and master_process:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })
        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
```
- 先得到lr并记录
	- 根据总共的`iter_num`获得学习率
	- `optimizer.param_groups`   对optimizer的各个参数进行分组管理，例如：分层，冻结参数，不同分层不同学习率，不同分层不同衰减
	- 这里把所有组的学习率都设为获得的`lr`
- 每间隔`eval_interval`次迭代和为主进程时，计算loss，记录日志和checkpoint
- 如果只是eval，计算完loss，保存后直接退出

**forward and backward**   
```python
for micro_step in range(gradient_accumulation_steps):
	if ddp:
		# in DDP training we only need to sync gradients at the last micro step.
		# the official way to do this is with model.no_sync() context manager, but
		# I really dislike that this bloats the code and forces us to repeat code
		# looking at the source of that context manager, it just toggles this variable
		model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
	with ctx:
		logits, loss = model(X, Y)
		loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
	# immediately async prefetch next batch while model is doing the forward pass on the GPU
	X, Y = get_batch('train')
	# backward pass, with gradient scaling if training in fp16
	scaler.scale(loss).backward()
```
- 如果是ddp，需要在一次累积梯度的最后一个step同步
	- 为什么是最后一个step?
	- 我们需要的是所有process上所有gradient_accumulation_steps个batch后累加的平均，总共应是 gradient_accumulation_steps * number_of_process(world_size) * batch_size 为一个大batch
	- 同步是加和平均，如果每次都同步，会多次平均得到错误结果
- 每个process每次循环计算一次loss和自己的梯度
	- 为什么loss要除以gradient_accumulation_steps
	- 保证梯度不会太大(梯度爆炸)
- 数据的获取是异步的，所以在前传的时候可以执行下一个batch数据的获取

```python
# clip the gradient
if grad_clip != 0.0:
	scaler.unscale_(optimizer)
	torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
```
- `unscale_(optimizer)` 将scaler放大loss导致放大的梯度还原
- `torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)` 
	- 裁剪所有参数的梯度范数(L2范数)，如果超过设定的阈值，梯度会按照比例 `阈值/实际值`整体缩放
	- 为什么要先还原再裁剪？后续使用`scaler`进行参数更新的时候会还原一次梯度，如果这里不先还原再裁剪，会导致`scaler`放大的倍数在这里先被除一次，导致两次缩小
	- 如果先缩放再更新`scaler.step(optimizer)`，因为已经使用过`unscale_`，会直接调用`optimizer`，不会再缩放一次

>关于混合精度计算的数据类型问题：
>如果`autocast`选择`float16`，前向传播和反向传播的矩阵计算会用`float16`计算，用于加速。此时反向计算梯度需要使用`scaler.scale(loss).backward()`来放大loss，保证不会出现梯度下溢
>但是放大后的梯度存在`float32`中，一直到`scaler.step(optimizer)`，会将放大的梯度缩放为原来大小

```python
# step the optimizer and scaler if training in fp16
scaler.step(optimizer)
scaler.update()
# flush the gradients as soon as we can, no need for this memory anymore
optimizer.zero_grad(set_to_none=True)
```
- 更新参数
- 释放梯度占用的内存


```python
# timing and logging
t1 = time.time()
dt = t1 - t0
t0 = t1
if iter_num % log_interval == 0 and master_process:
	# get loss as float. note: this is a CPU-GPU sync point
	# scale up to undo the division above, approximating the true total loss (exact would have been a sum)
	lossf = loss.item() * gradient_accumulation_steps
	if local_iter_num >= 5: # let the training loop settle a bit
		mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
		running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
	print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
iter_num += 1
local_iter_num += 1

# termination conditions
if iter_num > max_iters:
	break
```
- 记录时间和mfu
- 为什么`running_mfu`这么算？ 这是一个**指数移动平均(Exponential Moving Average, EMA)**，EMA对近期数据更敏感，更节省计算资源


```python
if ddp:
    destroy_process_group()
```
- 销毁进程组，释放资源

## `sample.py`

导入必要的库   
```python
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
from model import GPTConfig, GPT
```
- `pickle` 是python标准库中用于对象序列化与反序列化的模块，序列化指将对象转换为字节流，反序列化指从字节流中恢复对象。文件后缀名为`.pkl`  
- `tiktoken` 是OpenAI的tokenizer

超参数和全局变量   
```python
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 10 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
```
- 该文件用于生成，所以只能是resume或gpt2
- `start` 是起始符号
- `top_k` 使用top-k sample
- `bfloat16` BF16，是谷歌提出的用于深度学习的优化版FP16，FP16有1位符号位，5位指数位，10位尾数位，而BF16有8位指数位(与FP32相同)，7位尾数位(精度较低)。BF16范围更大，精度较低，需特定硬件支持
- `exec(open('configurator.py').read())` 修改参数，见`train.py`的解释

>##### 关于浮点数精度
>
>- 半精度是指float16(FP16)，1+5+10，内存占用小，可能有精度问题，GPU处理速度快
>- 单精度是指float32(FP32)，1+8+23，大多数框架的默认浮点数
>- 双精度是指float64(FP64)，1+11+52，用于高精度任务
>- bfloat16(BF16)，1+8+7，指数位(范围)与FP32相同，精度低于FP16，适用于与FP32无缝切换，需要硬件支持
>- TensorFloat32(TF32)，1+8+10，TF32是NVIDIA在Ampere架构GPU(A100，RTX30系列)引入的数据类型，指数位与FP32一致，精度与半精度一致，避免FP16的溢出问题，使用tensor core加速，与FP16速度接近。TF32虽然只有19位，但是占用32位内存空间

```python
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
```
- `torch.manual_seed()` 
- `torch.manual_seed()` 设置了CPU的随机种子，`torch.cuda.manual_seed()` 设置了GPU的随机种子
- `torch.backends.cuda.matmul.allow_tf32 = True`和`torch.backends.cudnn.allow_tf32 = True` 显式启用TF32加速，保证不同环境的一致性(Ampere架构中默认启用)


>##### 关于随机种子
>一个深度学习程序中可能需要设置多种随机种子
>- CPU 使用 `torch.manual_seed(seed)`
>- GPU使用 `torch.cuda.manual_seed(seed)`，同时每个GPU有自己的种子生成器，如果要设置所有GPU则使用 `torch.cuda.manual_seed_all(seed)`
>- 单纯设置上述两种种子，有些GPU操作仍存在不确定性，如cuDNN的卷积，需额外设置`torch.backends.cudnn.deterministic = True` 强制使用确定性算法，`torch.backends.cudnn.benchmark = False` 关闭自动优化
>- 如果使用了`numpy`和`random`，需要额外设置`random.seed(seed)`，`np.random.seed(seed)`

加载模型  
```python
# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, 'ckpt.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)
```
- 解释见`train.py`

```python
# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)
```
- `meta.pkl` 存储着处理数据时保存的encoder，decoder，在这里可能是教程中的character tokenizer
- 如果没有的话就默认tiktoken

```python
# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
```
- 如果起始的prompt来自文件，则先读取prompt
- encode prompt作为输入x

sample   
```python
# run generation
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            print(decode(y[0].tolist()))
            print('---------------')
```
- 注意decode的时候需要将输出转换为 整数列表
- tiktoken工作在CPU上














































































