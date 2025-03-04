import math
import inspect
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F
from einops import einsum, rearrange
from torch import Tensor

# # DEBUG -----------------------------------------
# import os
# import pickle
# meta_path = r"G:\project\nanoGPT\nonLlama\data\shakespeare_char\meta.pkl"
# if os.path.exists(meta_path):
#     with open(meta_path, 'rb') as f:
#         meta = pickle.load(f)
# dec = meta["itos"]
# decode = lambda s: "".join([dec[t] for t in s])
# # END DEBUG ------------------------------------

# LayerNorm with an optional bias
class LayerNorm(nn.Module):
    """LayerNorm but with an optional bias, copy from great work of karpathy"""
    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None
        
    def forward(self, input):
        return F.layer_norm(input, 
                            normalized_shape=self.weight.shape, 
                            weight=self.weight, 
                            bias=self.bias, 
                            eps=1e-5)

# use LlamaRMSNorm default
#TODO implement interface to use various norm type
class LlamaRMSNorm(nn.Module):
    """
    RMSNorm discard centralization for compute efficiency
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.var_eps = eps
        
    def forward(self, hidden_states: Tensor):
        input_type = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        var = hidden_states.pow(2).mean(dim=-1, keepdim=True)
        hidden_states = (hidden_states * torch.rsqrt(var + self.var_eps)).to(input_type)
        
        return self.weight * hidden_states


# Config
@dataclass
class LlamaConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT2 vocacab_size, depends on tokenizer(tiktoken here)
    n_embd: int = 768
    n_head: int = 16
    n_layer: int = 8
    n_kv_head: int = 4
    dropout: float = 0.0
    bias: bool = False
# Attention

# Attention block
#TODO GQA+RoPE 
# references https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py
# and https://github.com/naklecha/llama3-from-scratch

class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding"""
    def __init__(self, dim, base=10000):
        super().__init__()
        self.register_buffer("inv_freq", 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim)))
        self.seq_len_cached = None
        self.cos_cached = None
        self.sin_cached = None
    
    def forward(self, x, len_dim=1): # x has shape (B T C)
        seq_len = x.shape[len_dim]
        if seq_len != self.seq_len_cached:
            self.seq_len_cached = seq_len
            t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq) # freqs[i,j] = t[i] * inv_freq[j]
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)
            self.sin_cached = emb.sin()
            self.cos_cached = emb.cos()
        return self.cos_cached, self.sin_cached

def rotate_half(x):
    
    x1 = x[..., :x.shape[-1]//2]
    x2 = x[..., x.shape[-1]//2:]
    
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    # q/k has shape (b, h, n, d)
    # cos/sin has shape (n, d)
    return (q * cos + (rotate_half(q) * sin)), (k * cos + (rotate_half(k) * sin))


class LlamaAttention(nn.Module):
    
    def __init__(self, config: LlamaConfig):
        super().__init__()
        
        self.config = config
        assert(config.n_embd % config.n_head == 0 and config.n_head % config.n_kv_head == 0)
        self.head_dim = config.n_embd // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_kv_groups = config.n_head // config.n_kv_head # independent qkv as a group, the same as the number of head shared kv
        self.dropout = config.dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # attn
        self.q_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.k_proj = nn.Linear(config.n_embd, config.n_kv_head*self.head_dim, bias=config.bias)
        self.v_proj = nn.Linear(config.n_embd, config.n_kv_head*self.head_dim, bias=config.bias)
        
        self.o_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        # Flash Attention with GQA is an experimental feature, see https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        # self.flash = False  # Test manually GQA
        if not self.flash:
            # Maybe, I can't find which version exactly, the PR https://github.com/pytorch/pytorch/commit/8bc5ef563eab08bfa9e48c6a546aba994ab7828c
            print("WARNING: FlashAttention with GQA requires PyTorch >= 2.5.1.\n Using slow attention instead")
            # mask
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x, positional_embedding):
        B, T, C = x.size() # batch, time, channel(n_embd)
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim).transpose(1, 2)
        cos, sin = positional_embedding
        q, k = apply_rope(q, k, cos, sin)

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True,
                                                                 dropout_p= self.dropout if self.training else 0)
            # out has shape (b, h, n, d) 
            out = out.transpose(1, 2)
        else:
            #implementation references:
            # TODO check for correctness
            # https://medium.com/@maxshapp/grouped-query-attention-gqa-explained-with-code-e56ee2a1df5a
            # https://github.com/fkodom/grouped-query-attention-pytorch
            # https://github.com/naklecha/llama3-from-scratch/tree/main
            # https://iiosnail.blogspot.com/2024/10/einsum.html
            # TODO a blog on gqa implementation
            # TLDR: 
            #   for b, g, h, n, s, d:
            #   attn[b, g, h, n, s] += q[b, g, h, n, d] * k[b, h, s, d]
            q = rearrange(q, "b (h g) n d -> b g h n d", g=self.n_kv_groups)
            attn = einsum(q, k, "b g h n d, b h s d -> b g h n s") * (1.0 / math.sqrt(q.size(-1)))
            attn = attn.masked_fill(self.bias[None, :, :, :T, :T] == 0, float('-inf'))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = einsum(attn, v, "b g h n s, b h s d -> b g h n d")
            out = rearrange(out, "b g h n d -> b n (g h) d")
        
        out = out.contiguous().view(B, T, C)
        out = self.resid_dropout(self.o_proj(out))
    
        return out
    

# forward

# MLP layer
class MLP(nn.Module):
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config
        self.c_fc = nn.Linear(config.n_embd, 4*config.n_embd, bias=config.bias) # trick from GPT(see karparthy's awsome work)
        self.act_fn = nn.GELU()
        self.c_proj = nn.Linear(4*config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        x = self.c_proj(self.act_fn(self.c_fc(x)))
        x = self.dropout(x)
        return x

# Transformer block
class LlamaDecoderLayer(nn.ModuleDict):
    
    def __init__(self, config:LlamaConfig):
        super().__init__()
        self.config = config
        self.n_embd = config.n_embd
        self.attn = LlamaAttention(config)
        self.mlp = MLP(config)
        self.ln1 = LlamaRMSNorm(config.n_embd)
        self.ln2 = LlamaRMSNorm(config.n_embd)
    
    def forward(self, x, positional_embedding):
        # residual connection
        x = x + self.attn(self.ln1(x), positional_embedding)
        x = x + self.mlp(self.ln2(x)) 
        
        return x


# nonLlama
# init+forward+resume+optimizer+generate
# utils:get_num, estimate_mfu

class NonLlama(nn.Module):
    
    def __init__(self, config:LlamaConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.n_embd)
        self.rotary_emb = RotaryEmbedding(config.n_embd//config.n_head)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([LlamaDecoderLayer(config) for _ in range(config.n_layer)])
        self.ln_f = LlamaRMSNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        # weight tying
        self.embed_tokens.weight = self.lm_head.weight
        
        self.apply(self._init_weights)
        
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6))
        

    def forward(self, input_ids, targets=None):
        
        b, t = input_ids.size()
        assert t <= self.config.block_size, f"sequence of length {t} larger than block_size, block size is only {self.config.block_size}"
        
        hidden_states = self.embed_tokens(input_ids)
        positional_embedding = self.rotary_emb(hidden_states)
        
        x = self.dropout(hidden_states)
        
        for layer in self.layers:
            x = layer(x, positional_embedding)
            
        x = self.ln_f(x)
        
        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            
        else:
            
            logits = self.lm_head(x[:, [-1], :]) # only the last timestep, use list [-1] preserve the time dim. e.g.(B, 1, C)
            loss = None
            
        return logits, loss
    
    # utils
    def get_num_params(self, non_embedding=True):
        """
        return the number of parameters in the model 
        copy from nanoGPT https://github.com/karpathy/nanoGPT
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.embed_tokens.weight.numel()
        return n_params
    
    def _init_weights(self, module):
        """ 
        Initialize weights in the model 
        copy from nanoGPT https://github.com/karpathy/nanoGPT
        """
        
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)    
            
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        """ 
        optimizer copy from nanoGPT https://github.com/karpathy/nanoGPT
        """
        
        param_dict = {pn: p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad} # filter out those that do not require grad
        
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2] # find bias params
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}, # bias does not require regularization
        ]
        
        num_decay_params = sum(p.numel() for p in decay_params)
        num_non_decay_params = sum(p.numel() for p in nodecay_params)
        
        print(f"num of decayed params tensors: {len(decay_params)}, with {num_decay_params} params")
        print(f"num of non-decayed params tensors: {len(nodecay_params)}, with {num_non_decay_params} params")
        
        # AdamW optimizer, use fused if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters # check signature
        use_fused = fused_available and device_type == "cuda"
        extra_args = dict(fused=True) if use_fused else dict() # fused argument here
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        
        return optimizer
        
        
    def estimate_mfu(self, fwdbwd_per_iter, dt, gpu):
        gpu_peak_flops = {
            "A100": 312e12,
            "A800": 78.2e12,
            "RTX-3080": 29.77e12, # RTX-3080 does not support bfloat16
            "vGPU-32GB": 103e12, # I don't know what it is
        }
        
        if gpu not in gpu_peak_flops:
            print(f"not support {gpu}, not friendly at all")
            return 0
        
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        
        flops_per_token = 6*N + 12*L*H*Q*T # 关于这个LHQT怎么来的我不是很明白，可能是3次前后传运算，2次矩阵相乘，2次矩阵运算(乘加)
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = gpu_peak_flops[gpu]
        
        mfu = flops_achieved / flops_promised
        
        return mfu
    
    @torch.no_grad
    def generate(self, input_ids, max_new_tokens, temperature=1.0, top_k=None):
        """  
        input_ids has shape (B, T)
        """
        
        for _ in range(max_new_tokens):
            
            input_window_size = self.config.block_size
            input_ids_trunc = input_ids if input_ids.size(1) <= input_window_size else input_ids[:, -input_window_size:]
            
            logits, _ = self(input_ids_trunc)
            logits = logits[:, -1, :] / temperature
            
            if top_k is not None:
                val, ind = torch.topk(logits, min(top_k, logits.shape[-1]))
                logits[logits < val[:, [-1]]] = float('-inf')
                
            probs = F.softmax(logits, dim=-1)
            
            next_id = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat((input_ids, next_id), dim=-1)
            
        return input_ids

if __name__ == "__main__":
    config = LlamaConfig()
    print(config.n_embd)
    
    model = NonLlama(config)
    optimizer = model.configure_optimizers(1e-5, 1e-5, (0.1, 0.1),"cuda")
    logits, loss = model(torch.ones(1, 4, dtype=torch.long), torch.ones(1,4, dtype=torch.long)+torch.ones(1,4, dtype=torch.long))
    loss.backward()
    optimizer.step()
            
            
            