## References

[Let's build GPT: from scratch, in code, spelled out. - YouTube](https://www.youtube.com/watch?v=kCc8FmEb1nY)   

## Reading and exploring the data

tokenizer map the characters/words to integer/code  
In character size tokenizer the len of chars in dataset is the vocabulary size of model   
so with character size tokenizer, we have small size of code book but long sequence of encode   
![[Pasted image 20250110160507.png]]  
#### tokenizer
nanoGPT use the very simple tokenizer   
```python
stoi = {ch:i for ch, i in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[i] for i in s]
decode = lambda l: "".join([itos[i] for i in l])
``` 

> 感觉，这里用lambda 主要是为了省行数，实际写的时候可以改一下

#### data split

Encode all the data(shakespere works here)   
split training dataset and validation dataset.   
0.9 as training data, and the rest as validation data   

> 尽管这只是一个样例，但是莎士比亚的作品如果是有序的，这么划分有点奇怪  

training on chunks of text    
the chunks of text size `block_size` is also nemed context size   
```python
x = train_data[:block_size]
y = train_data[1:block_size+1]  # one offset on target
```

**注意y比x向右偏移了1位，下一位才是target**   

The input shoulbe be a sequence of context, and the output is the next token    
```python
for t in range(block_size):
	context = x[:t+1]
	target = y[t]
	print(f"when the input is {context}, the output should be {target}")
```

> 再次注意输入**从1位到整个上下文长度**(`block_size`)    
> 这样transformer能够从一位开始predict直到block_size  
> 本身的输入最长就是block_size，所以一直到整个长度作为输入进行训练也很合理

Consider the batch dimension    
```python
def get_batch(split):
	data = train_data if split == "train" else val_data
	idx = torch.randint(len(data) - block_size, (batch_size))
	x = torch.stack([data[i:block_size] for i in idx])
	y = torch.stack([data[i+1:block_size+1] for i in idx])
	return x, y
```

> 个人感觉是很巧妙的实现，随机选取idx获得block，这何尝不是一种蒙特卡洛  
> 但是这样应该没办法确保所有数据都被使用，真实的batch是如何实现的 👈看看d2l

得到的x，y是相对应的(4, 8)的二维数组   
注意这对应了32对训练数据(every position, independent as far as the transformer concerned)       
32对数据是一个batch size 为4的一个batch的输入，训练时对transformer来说相当于一个batch为32(32对数据计算loss)，这是通过掩码实现的   

## Model

### Build BigramLanguageModel
using pytorch for efficiency  

```python
class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

	def forward(self, idx, targets):
		logits = self.token_embedding_table(idx) # B, T, C
		B, T, C = logits.shape
		logits = logits.view(B*T, C)
		targets = targets.view(B*T)
		loss = F.cross_entropy(logits, targets)  # use nn.functional 

		return logits, loss
```

#### Some Supplement

##### What is BigramLanguageModel?  

Generally speaking, bigram language model is the model that predict next token only depend on the last token (or words pair)   
`BigramLanguageModel` here performs as a baseline   
more details see [[makemore]] 👈 **待补**    

##### Why reshape?

输入是一个 `B, T` 的向量，B是`batch_size`, T是time step      
经过embedding后，logits是一个`B, T, C` 的向量，其中C是channel，即`vocab_size`   
`torch.nn.Cross_Entropy` (a class by the way) 接受`B, C` 的输入，要求channel在第二位   
> `torch.nn.cross_entropy`  is the functional version

So reshape is needed  
What we care about is the loss of the whole input(batch), so we directly change the size into `B*T`, make the batch size for cross-entropy become all input(32 rows here)   
##### What does the `cross-entropy` do

~~到这里就很难假装会英语了~~    
**数学上**，Cross-entropy 写作   

$$
\mathrm{C r o s sE n t r o p y}=-\sum_{i=1}^{C} y_{i} \operatorname{l o g} ( p_{i} ) 
$$
其中，$y_i$是目标分布的概率，$p_i$是预测分布的概率   
当用于类别计算时，目标分类表示为`[0, 1, 2]`，即第一个样本为0类，第二个为1类，以此类推   
输入是一个二维数组，每一个样本是在各个类别的未归一化的**logits**    
```
[[0.2, 2.0, 1,2],
[0.3, 3.0, 0.4],
[0.6, 1.0, 1.5]]
```

目标的分布，实际上只在正确样本上概率为1，其余都为0   
```
[[0, 1, 0],
[0, 1, 0],
[0, 0, 1]]
```

所以在分类任务计算cross-entropy，只需要计算正确类上的 **negative log likelihood(NLL)**   

$$
\mathrm{N L L}=-\operatorname{l o g} ( p_{\mathrm{c o r r e c t}} ) 
$$

如果多个样本/(batch size here)   
那就是log likelihood 累加再平均      

$$
\mathrm{B a t c h} \; \mathrm{L o s s}=\frac{1} {N} \sum_{n=1}^{N} \mathrm{L o s s}_{n} 
$$

**代码上**    
`cross_entropy`内置softmax，输入未归一化的`logits`和一一对应的integer形式的`labels`  
- 进行softmax之后
- 查找对应的正确类别计算NLL，
- 累加后取平均   


#### Generate next tokens

refine the model   
```python
class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

	def forward(self, idx, targets = None):
		logits = self.token_embedding_table(idx) # B, T, C
		
		if targets in None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)  # use nn.functional 
			
		return logits, loss

	def generate(self, idx, max_new_tokens):
		# idx here is (B,T) array of indices in the current context
		for _ in range(max_new_tokens):
			logits, loss = self(idx) # logits here will be a (B, T, C) array since no target provided
			logits = logits[:, -1, :] # only care about the final token (B, C)
			probs = F.softmax(logits, dim=-1) # (B, C)
			idx_next = torch.multinomial(probs, num_samples=1) # sample next token (B, 1)
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx
```

- We introduce `generate` function to generate `next_max_tokens`   
- `self(x)` call the forward function and return the logits(ignoring the loss)
- `torch.multinomial` is a function that sample idx according to `weight`(probs)  

**注意这里是按batch生成的`next token`**   

```python
idx = torch.zeros((1,1), dtype=torch.long)
decode(model.generate(idx, 100)[0].tolist())
```

#### Train the model

```python
optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)
```

Karpathy said `AdamW` works well in setting that the learning rate is 3e-4.  Since the model is small, use 1e-3 instead(set much higher)    

```python
batch_size = 32
for step in range(100):
	xb, yb = get_batch("train")
	logits, loss = m(xb, yb)
	loss.backward()
	optimizer.step()
	print(loss.item())

print(loss.item())
```

> In the video, maybe 10000 steps works

#### `time.sleep(5)` 

到这里是一些简单的实现   
已经实现的内容如下：  
- 一个简易的character size tokenizer
- 一个按随机数获得batch的dataloader
- 一个baseline model  Bigram language model

但是存在几个问题  
- 首先batch是随机选取的，并不遍历整个数据集，这导致了很大的随机性(noise)
	- 减少这些noise, Karpathy在外面套上一层循环，即累计一定次数的batch，当循环结束后计算平均loss，将此作为一次iteration  👈 有无可能自己写一个简单的dataloader
- 现在的计算是在CPU上，需要转换到GPU

> 出现了一个`@torch.no_grad()` 指示当前函数不需要梯度(torch不需要维护梯度状态)，加速计算的技巧   


### Self-attention block

#### Mathematical trick in self-attention

We need the current token to conmunicate with the previous tokens   
The simplest way is average   

For intuitive, a simple program can be writen as   
```python
x = torch.rand
xbow = torch.zeros((B, T, C))
for b in range(B):
	for t in range(T):
		xprev = x[b, :t+1] # (t, C)
		xbow[b, t] = torch.mean(xprev, 0)
```

> `xbow` means bag of words   

`xbow` contains the mean of previous tokens(include the current one) at every position  

But the implementation is not efficient enough, using matrix multiplication instead of traversing in two loop    

首先，矩阵相乘$A \times B$ 有两个角度，一个是结果$C$第$i$行是B每一行依据A的第$i$行加权求和的结果(行的角度)，一个是C的第$i$列，是A的每一列依据B的第$i$列加权求和的结果   

从行的角度来说  
$$
c_{ij} = \mathbf{a}_i \cdot \mathbf{b}_j
$$
即a的第i行与a的第j列的点乘   

因此，利用这样的特性，可以直接对token进行累加和平均   

输入的token，一个批次为矩阵 B $(T, C)$，我们需要时间维度上的均值，即在行角度累加并求平均。只需要将A设置为1的下三角矩阵，我们就能得到矩阵B行上的累加结果   
引入归一化，则只需要让下三角矩阵的每一行除以这一行的和(即行数)   

```python
wei = torch.tril(torch.ones(T, T)) # 1 的下三角阵
wei = wei / wei.sum(1, keepdim=True) # 列角度加和并保持维数
xbow = wei @ x # (T, T) @ (B, T, C) broadcast补全Batch_size, (B,T,T) @ (B,T,C) --> (B,T,C)
```

> 可以用 `torch.allclose(xbow, xbow2)` 比较两种方法结果是否一致


A more readable version   
```python
tril = torch.tril(torch.ones(T, T))
wei = torch.zeros((T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)
xbow3 = wei @ x
```
这段代码的思路服务于self-attention  
- 首先得到下三角矩阵，表明不考虑未来信息
- 然后得到权重矩阵，初始化为0，但实际上当注意力被计算后会是其他数值，将未来信息掩盖成`-inf`  
- 经过softmax得到最终权重

#### Self-attention block   

Firstly we need to modify the bigram language model by adding linear head and positional embedding   

```python
class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.postional_embedding_table = nn.Embedding(block_size, n_embd)
		self.lm_head = nn.linear(n_embd, vocab_size)

	def forward(self, idx, targets = None):
		B, T = idx.shape
		token_embd = self.token_embedding_table(idx) # B, T, C
		pos_embd = self.positonal_embedding_table(torch.arange(T, device=device)) # T, C
		x = token_embd + pos_embd  # broadcast
		logits = self.lm_head(x)

		if targets in None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)  # use nn.functional 
			
		return logits, loss

	def generate(self, idx, max_new_tokens):
		# idx here is (B,T) array of indices in the current context
		for _ in range(max_new_tokens):
			logits, loss = self(idx) # logits here will be a (B, T, C) array since no target provided
			logits = logits[:, -1, :] # only care about the final token (B, C)
			probs = F.softmax(logits, dim=-1) # (B, C)
			idx_next = torch.multinomial(probs, num_samples=1) # sample next token (B, 1)
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx
```


And then we implement the self-attention block   
```python
head_size = 16
query = torch.nn.linear(n_embd, head_size, bias = False)
key = torch.nn.linear(n_embd, head_size, bias = False)
value = torch.nn.linear(n_embd, head_size, bias = False) # bias = False 使linear head 只用于改变维度
# Q K V
Q = query(x) # B, T, head_size
K = key(x)  # B, T, head_size
V = value(x) # B, T, head_size
# attention matrix
wei = Q @ K.tranpose(-2, -1)  # B, T, T
# masked
tril = torch.tril(torch.ones(T, T))
wei = wei.masked_fill(tril == 0, float('-inf'))
wei = F.softmax(wei, dim=-1)  # B, T, T
# final output
out = wei @ V
```
- 如果是encoder block，则没有`tril`的部分(不需要mask)
- 此时的attention没有除以 $\sqrt{d_k}$ (`head_size`)/还不是scaled self-attention

##### 为什么要Scaled

减小方差   
softmax是一个归一化函数，如果进行归一的数据差距过大，容易导致函数收敛到极端(1)  
即，注意力/token之间的相关性只在个别数值大的position有效。      
通过scaled，除以$\sqrt{d_k}$来减小方差     
[[Why dividing by square root of dimension of key vector]]   

##### Attention Head

```python
class Head(nn.Module):

	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias = False)
		self.query = nn.Linear(n_embd, head_size, bias = False)
		self.value = nn.Linear(n_embd, head_size, bias = False)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x)  # (B, T, head_size)
		q = self.query(x) # (B, T, head_size)
		v = self.value(x) # (B, T, head_size)
		# compute attention score ("affinities")
		wei = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		wei = F.softmax(wei, dim=-1) # (B, T, T)

		v = value(x)
		out = wei @ v # (B, T, head_size)
		return out
```
- triangular module is not a parameter module of the model. In pytorch, it should be assign with `register_buffer`   

```python
class MultiHeadAttention(nn.Module):

	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

	def forward(self, x):
		return torch.cat([h(x) for h in self.heads], dim = -1)
```
- concatenate over channel dimension

##### FeedForward layer

```python
class FeedForward(nn.Module):
	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, n_embd),
			nn.ReLU(),
		)
```

Add computation layers   
可以认为是帮助获得注意力之后的token去思考注意力的影响    
> 这里的实现将batch和Time都当成是Batch，对每一个token都进行计算    

##### Attention block

![[Pasted image 20250117165015.png]]    


```python
class Block(nn.Module):
	def __init__(self, n_embd, n_head):
		super()._init__()
		head_size = n_embd // n_head
		self.mha = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embd)
	def forward(self, x):
		x = self.mha(x)
		x = self.ffwd(x)
		return x
```


```python

class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.postional_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(
			Block(n_embd, n_head=4),
			Block(n_embd, n_head=4),
			Block(n_embd, n_head=4),
		)
		self.lm_head = nn.linear(n_embd, vocab_size)
		

	def forward(self, idx, targets = None):
		B, T = idx.shape
		token_embd = self.token_embedding_table(idx) # B, T, C
		pos_embd = self.positonal_embedding_table(torch.arange(T, device=device)) # T, C
		x = token_embd + pos_embd  # broadcast
		x = self.blocks(x)
		logits = self.lm_head(x)
		
		if targets in None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)  # use nn.functional 
			
		return logits, loss

	def generate(self, idx, max_new_tokens):
		# idx here is (B,T) array of indices in the current context
		for _ in range(max_new_tokens):
			logits, loss = self(idx) # logits here will be a (B, T, C) array since no target provided
			logits = logits[:, -1, :] # only care about the final token (B, C)
			probs = F.softmax(logits, dim=-1) # (B, C)
			idx_next = torch.multinomial(probs, num_samples=1) # sample next token (B, 1)
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx
```

#### Tricks borrowed from the original paper

##### Residual connections

add resudual connection and some projections   
```python
class Block(nn.Module):
	def __init__(self, n_embd, n_head):
		super()._init__()
		head_size = n_embd // n_head
		self.mha = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embd)
	def forward(self, x):
		x = x + self.mha(x)
		x = x + self.ffwd(x)
		return x
```

```python
class MultiHeadAttention(nn.Module):

	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embd, n_embd)

	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim = -1)
		out = self.proj(out)
		return out
```

```python
class FeedForward(nn.Module):
	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, n_embd),
			nn.ReLU(),
			nn.Linear(n_embd, n_embd),
		)
```

And the original paper use 4 times embedding dimension 

```python
class FeedForward(nn.Module):
	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.ReLU(),
			nn.Linear(4 * n_embd, n_embd),
		)
```

#### Layer Norm & Drop out

Transformer add layer norm after the transformation, but for now it is common to apply layer norm before transformation  
We can add layer norm in   
- before transformation
- at the end of transformer

```python
class Block(nn.Module):
	def __init__(self, n_embd, n_head):
		super()._init__()
		head_size = n_embd // n_head
		self.mha = MultiHeadAttention(n_head, head_size)
		self.ffwd = FeedForward(n_embd)
		self.ln1 = nn.LayerNorm(n_embd)
		self.ln3 = nn.LayerNorm(n_embd)
	def forward(self, x):
		x = x + self.mha(ln1(x))
		x = x + self.ffwd(ln2(x))
		return x
```

```python

class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.postional_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(
			Block(n_embd, n_head=4),
			Block(n_embd, n_head=4),
			Block(n_embd, n_head=4),
			nn.LayerNorm(n_embd),
		)
		self.lm_head = nn.linear(n_embd, vocab_size)
```


Drop out can be added  
- right before the residual connection back
- after computing attention score(randomly prevent some of the nodes from communication)

```python
class MultiHeadAttention(nn.Module):

	def __init__(self, num_heads, head_size):
		super().__init__()
		self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
		self.proj = nn.Linear(n_embd, n_embd)
		self.dropout = nn.Dropout(dropout)
	def forward(self, x):
		out = torch.cat([h(x) for h in self.heads], dim = -1)
		out = self.proj(out)
		out = self.dropout(out)
		return out
```

```python
class FeedForward(nn.Module):
	def __init__(self, n_embd):
		super().__init__()
		self.net = nn.Sequential(
			nn.Linear(n_embd, 4 * n_embd),
			nn.ReLU(),
			nn.Linear(4 * n_embd, n_embd),
			nn.Dropout(dropout),
		)
```


```python
class Head(nn.Module):

	def __init__(self, head_size):
		super().__init__()
		self.key = nn.Linear(n_embd, head_size, bias = False)
		self.query = nn.Linear(n_embd, head_size, bias = False)
		self.value = nn.Linear(n_embd, head_size, bias = False)
		sefl.dropour = nn.Dropout(dropout)
		self.register_buffer('tril', torch.tril(torch.ones(block_size, block_szie)))

	def forward(self, x):
		B, T, C = x.shape
		k = self.key(x)  # (B, T, head_size)
		q = self.query(x) # (B, T, head_size)
		v = self.value(x) # (B, T, head_size)
		# compute attention score ("affinities")
		wei = q @ k.transpose(-2, -1) * C**(-0.5) # (B, T, T)
		wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
		wei = F.softmax(wei, dim=-1) # (B, T, T)
		wei = self.dropout(wei)
		v = value(x)
		out = wei @ v # (B, T, head_size)
		return out
```


#### Then scale up

```python

class BigramLanguageModel(nn.Module):

	def __init__(self, vocab_size):
		super().__init__()
		self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
		self.postional_embedding_table = nn.Embedding(block_size, n_embd)
		self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
		self.ln_f = nn.LayerNorm(n_embd)
		self.lm_head = nn.linear(n_embd, vocab_size)
		
	def forward(self, idx, targets = None):
		B, T = idx.shape
		token_embd = self.token_embedding_table(idx) # B, T, C
		pos_embd = self.positonal_embedding_table(torch.arange(T, device=device)) # T, C
		x = token_embd + pos_embd  # broadcast
		x = self.blocks(x)
		x = self.ln_f(x)
		logits = self.lm_head(x)
		
		if targets in None:
			loss = None
		else:
			B, T, C = logits.shape
			logits = logits.view(B*T, C)
			targets = targets.view(B*T)
			loss = F.cross_entropy(logits, targets)  # use nn.functional 
			
		return logits, loss

	def generate(self, idx, max_new_tokens):
		# idx here is (B,T) array of indices in the current context
		for _ in range(max_new_tokens):
			logits, loss = self(idx) # logits here will be a (B, T, C) array since no target provided
			logits = logits[:, -1, :] # only care about the final token (B, C)
			probs = F.softmax(logits, dim=-1) # (B, C)
			idx_next = torch.multinomial(probs, num_samples=1) # sample next token (B, 1)
			idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
		return idx
```

adjust batch_size, block_size, iteration times, n_embd, n_head, n_layer according to compute resources    

