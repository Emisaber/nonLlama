
# CONFIG ------------------------------------------------------
# Train on shakespeare dataset
# a test demo

out_dir = 'out-shakespeare-nonllama-demo'
eval_interval = 250 # evaluate once per eval_interval 
log_interval = 10
eval_iters = 50  # how many batch/iters when evaluate ?
eval_only = False
always_save_checkpoint = True
init_from = 'scratch'
num_proc_load_data = 2

# wandb logging
wandb_log = False

# data
dataset = 'shakespeare_char'
gradient_accumulation_steps = 2
batch_size = 64

# model
block_size = 256
n_embd = 384
n_head = 8
n_layer = 6
n_kv_head = 2
dropout = 0.0 # 0 when pre-training, fine-tuning try 0.1+ 
bias = False # no bias inside Linear layers

# optimizer
peak_learning_rate = 1e-3 
max_iters = 2000 # need adjusted
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients

# learning rate
decay_lr = True
warmup_iters = 100 # need adjusted, how many iterations to warm up for
lr_decay_iters = 5000 # per Chinchilla
min_lr = 1e-4



# system
device = 'cuda'
compile = False

